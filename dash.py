import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# ============================================================
# 0. Conexión a la BD
# ============================================================

user = "root"
password = "Levp13aa"
host = "localhost"
database = "futbol_dw"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# ============================================================
# 1. Constantes / helpers (Gold)
# ============================================================

BIG5_LEAGUES = ["Laliga", "Premier_League", "Serie_A", "Bundesliga", "Ligue_1"]

# Mapa simple rol -> position_clean en silver_players
POSITION_GROUPS = {
    "portero": ["GK"],
    "defensa": ["DF"],
    "mediocampista": ["MF"],
    "delantero": ["FW"],
}

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def standardize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd

# ============================================================
# 2. Perfil del equipo (silver_teams -> vector estilo)
# ============================================================

def get_team_profile(engine, team_name: str, season: str) -> tuple[dict, np.ndarray]:
    query = """
        SELECT
            team_name,
            league,
            season,
            AVG(idx_possession_style)     AS pos_style,
            AVG(idx_verticality_style)    AS vert_style,
            AVG(idx_offensive_style)      AS off_style,
            AVG(idx_defensive_style)      AS def_style,
            AVG(idx_aggressiveness_style) AS aggr_style
        FROM silver_teams
        WHERE team_name = %(team_name)s
          AND season = %(season)s
        GROUP BY team_name, league, season;
    """
    df = pd.read_sql(query, engine, params={"team_name": team_name, "season": season})
    if df.empty:
        raise ValueError(f"No se encontró perfil de equipo para {team_name} en {season}")
    row = df.iloc[0].to_dict()
    x_team = np.array([
        row.get("pos_style", np.nan),
        row.get("vert_style", np.nan),
        row.get("off_style", np.nan),
        row.get("def_style", np.nan),
        row.get("aggr_style", np.nan),
    ], dtype=float)
    x_team = np.nan_to_num(x_team, nan=0.0)
    return row, x_team

# ============================================================
# 3. Pool de jugadores (silver_players -> 1 fila por jugador)
# ============================================================

def get_players_pool(
    engine,
    season: str,
    leagues: list[str] | None = None,
    leagues_big5: bool = True,
    min_minutes: int = 600,
) -> pd.DataFrame:
    query = """
        SELECT
            player_name_clean         AS player_name,
            team_name,
            league,
            season,
            position_clean            AS position,
            SUM(minutes_played)       AS minutes_total,
            AVG(idx_finishing)        AS finishing,
            AVG(idx_playmaking)       AS playmaking,
            AVG(idx_progression)      AS progression,
            AVG(idx_involvement)      AS involvement,
            AVG(idx_defending)        AS defending,
            AVG(idx_discipline)       AS discipline,
            AVG(
                CASE
                    WHEN player_age IS NULL THEN NULL
                    ELSE CAST(SUBSTRING_INDEX(player_age, '-', 1) AS SIGNED)
                END
            ) AS age_years
        FROM silver_players
        WHERE season = %(season)s
        GROUP BY
            player_name_clean, team_name, league, season, position_clean
        HAVING minutes_total >= %(min_minutes)s;
    """
    df = pd.read_sql(query, engine, params={"season": season, "min_minutes": min_minutes})

    if leagues_big5:
        df = df[df["league"].isin(BIG5_LEAGUES)]
    elif leagues is not None:
        df = df[df["league"].isin(leagues)]

    df = df.reset_index(drop=True)

    for col in ["minutes_total", "finishing", "playmaking", "progression",
                "involvement", "defending", "discipline", "age_years"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ============================================================
# 4. Proyección jugador -> espacio de equipo
# ============================================================

def project_players_to_team_space(df_players: pd.DataFrame) -> pd.DataFrame:
    df = df_players.copy()
    df["style_pos"] = df[["involvement", "playmaking"]].mean(axis=1, skipna=True)
    df["style_vert"] = df["progression"]
    df["style_off"] = df[["finishing", "playmaking"]].mean(axis=1, skipna=True)
    df["style_def"] = df["defending"]
    df["style_aggr"] = df["defending"]  # intens/similar

    style_cols = ["style_pos", "style_vert", "style_off", "style_def", "style_aggr"]
    df[style_cols] = df[style_cols].fillna(0.0)
    return df

# ============================================================
# 5. Vector objetivo del equipo (real + entrenador)
# ============================================================

def build_team_target_vector(
    x_team_real: np.ndarray,
    x_coach: np.ndarray | None = None,
    lam: float = 0.5,
) -> np.ndarray:
    x_team_real = np.asarray(x_team_real, dtype=float)
    if x_coach is None:
        return x_team_real
    x_coach = np.asarray(x_coach, dtype=float)
    if x_coach.shape != x_team_real.shape:
        raise ValueError(f"x_coach debe tener shape {x_team_real.shape}, recibió {x_coach.shape}")
    return (1 - lam) * x_team_real + lam * x_coach

# ============================================================
# 6. Recomendador principal
# ============================================================

def recommend_players_for_team(
    engine,
    team_name: str,
    season: str,
    x_ideal_role: list[float] | np.ndarray,
    *,
    leagues: list[str] | None = None,
    leagues_big5: bool = True,
    min_minutes: int = 600,
    role: str | None = None,
    position_whitelist: list[str] | None = None,
    x_coach: list[float] | np.ndarray | None = None,
    lam_coach: float = 0.5,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.1,
    top_n: int = 30,
) -> pd.DataFrame:

    # 1) Perfil equipo
    team_row, x_team_real = get_team_profile(engine, team_name, season)

    x_ideal_role = np.asarray(x_ideal_role, dtype=float)
    if x_ideal_role.shape != x_team_real.shape:
        raise ValueError(
            f"x_ideal_role debe tener shape {x_team_real.shape}, recibió {x_ideal_role.shape}"
        )

    x_team_target = build_team_target_vector(
        x_team_real=x_team_real,
        x_coach=np.asarray(x_coach, dtype=float) if x_coach is not None else None,
        lam=lam_coach,
    )

    need_vec = x_ideal_role - x_team_target

    # 2) Pool jugadores
    df_players = get_players_pool(
        engine=engine,
        season=season,
        leagues=leagues,
        leagues_big5=leagues_big5,
        min_minutes=min_minutes,
    )
    if df_players.empty:
        raise ValueError("No se encontraron jugadores en el pool con esos filtros.")

    # Filtrar por rol / posición
    allowed_positions = None
    if position_whitelist is not None:
        allowed_positions = position_whitelist
    elif role is not None and role in POSITION_GROUPS:
        allowed_positions = POSITION_GROUPS[role]

    if allowed_positions is not None and "position" in df_players.columns:
        df_players = df_players[df_players["position"].isin(allowed_positions)].copy()

    if df_players.empty:
        raise ValueError("No hay jugadores tras filtrar por posición/rol.")

    # 3) Proyección a espacio de equipo
    df_players = project_players_to_team_space(df_players)
    style_cols = ["style_pos", "style_vert", "style_off", "style_def", "style_aggr"]

    x_team_target_vec = np.asarray(x_team_target, dtype=float)
    need_vec = np.asarray(need_vec, dtype=float)

    def _style_fit_row(row):
        v = row[style_cols].to_numpy(dtype=float)
        return cosine_sim(v, x_team_target_vec)

    def _needs_fit_row(row):
        v = row[style_cols].to_numpy(dtype=float)
        return float(np.dot(need_vec, v))

    df_players["style_fit_raw"] = df_players.apply(_style_fit_row, axis=1)
    df_players["needs_fit_raw"] = df_players.apply(_needs_fit_row, axis=1)

    # 4) Edad & disciplina
    if "age_years" in df_players.columns:
        age = pd.to_numeric(df_players["age_years"], errors="coerce")
        age_min, age_max = age.min(), age.max()
        if np.isfinite(age_min) and np.isfinite(age_max) and age_max > age_min:
            df_players["age_potential_raw"] = (age_max - age) / (age_max - age_min)
        else:
            df_players["age_potential_raw"] = 0.0
    else:
        df_players["age_potential_raw"] = 0.0

    df_players["discipline_raw"] = -pd.to_numeric(
        df_players["discipline"], errors="coerce"
    ).fillna(0.0)

    # 5) Normalizar componentes y score final
    df_players["style_fit_z"] = standardize(df_players["style_fit_raw"])
    df_players["needs_fit_z"] = standardize(df_players["needs_fit_raw"])
    df_players["age_potential_z"] = standardize(df_players["age_potential_raw"])
    df_players["discipline_z"] = standardize(df_players["discipline_raw"])

    df_players["score"] = (
        alpha * df_players["style_fit_z"]
        + beta * df_players["needs_fit_z"]
        + gamma * df_players["age_potential_z"]
        + delta * df_players["discipline_z"]
    )

    df_players = df_players.sort_values("score", ascending=False).reset_index(drop=True)

    cols_out = [
        "player_name", "team_name", "league", "position",
        "minutes_total",
        "style_fit_raw", "needs_fit_raw",
        "age_years",
        "discipline",
        "score",
    ]
    cols_out = [c for c in cols_out if c in df_players.columns]

    return df_players[cols_out].head(top_n), team_row, x_team_real, x_team_target

# ============================================================
# 7. UI Streamlit
# ============================================================

st.set_page_config(page_title="Recomendador de jugadores", layout="wide")

st.title("⚽ Sistema de recomendación de jugadores (encaje e impacto)")

st.markdown(
    """
Este dashboard te permite, como entrenador/director deportivo:

- Elegir tu **equipo** y temporada.
- Definir cómo quieres que juegue el equipo (estilo).
- Definir el **perfil ideal** del rol a reforzar.
- Obtener una lista de jugadores que:
  - Encajan con tu estilo de juego (style_fit).
  - Corrigen las carencias actuales (needs_fit).
  - Tienen buena edad/potencial y disciplina.
"""
)

# -------- Sidebar: filtros básicos --------
st.sidebar.header("1. Filtros generales")

# Temporadas disponibles
df_seasons = pd.read_sql("SELECT DISTINCT season FROM silver_teams;", engine)
seasons = sorted(df_seasons["season"].dropna().unique())
season = st.sidebar.selectbox("Temporada", seasons)

# Ligas disponibles
df_leagues = pd.read_sql("SELECT DISTINCT league FROM silver_teams;", engine)
all_leagues = sorted(df_leagues["league"].dropna().unique())

solo_big5 = st.sidebar.checkbox("Solo 5 grandes ligas", value=True)
if solo_big5:
    leagues_sel = [l for l in all_leagues if l in BIG5_LEAGUES]
else:
    leagues_sel = st.sidebar.multiselect("Ligas", options=all_leagues, default=all_leagues)

# Equipos según temporada + ligas
if solo_big5:
    query_teams = """
        SELECT DISTINCT team_name
        FROM silver_teams
        WHERE season = %(season)s
          AND league IN %(leagues)s
        ORDER BY team_name;
    """
    df_teams = pd.read_sql(query_teams, engine, params={"season": season, "leagues": tuple(BIG5_LEAGUES)})
else:
    query_teams = """
        SELECT DISTINCT team_name
        FROM silver_teams
        WHERE season = %(season)s
          AND league IN %(leagues)s
        ORDER BY team_name;
    """
    df_teams = pd.read_sql(query_teams, engine, params={"season": season, "leagues": tuple(leagues_sel)})

teams = df_teams["team_name"].tolist()
team_name = st.sidebar.selectbox("Equipo", teams)

role_ui = st.sidebar.selectbox(
    "Rol a reforzar",
    options=["(sin filtro)", "Portero", "Defensa", "Mediocampista", "Delantero"],
)
role_param = None if role_ui.startswith("(") else role_ui.lower()

min_minutes = st.sidebar.number_input("Minutos mínimos jugados", min_value=0, max_value=5000, value=600, step=100)
top_n = st.sidebar.number_input("Número de jugadores a recomendar", min_value=5, max_value=100, value=30, step=5)

# -------- Sidebar: estilo ideal del rol --------
st.sidebar.header("2. Estilo ideal del rol (x_ideal_role)")

st.sidebar.markdown("Valores entre -2 (muy bajo) y 2 (muy alto).")

role_pos = st.sidebar.slider("Posesión / elaboración (rol)", -2.0, 2.0, 0.0, 0.1)
role_vert = st.sidebar.slider("Verticalidad (rol)", -2.0, 2.0, 0.5, 0.1)
role_off = st.sidebar.slider("Ataque / finalización (rol)", -2.0, 2.0, 1.0, 0.1)
role_def = st.sidebar.slider("Defensa (rol)", -2.0, 2.0, 0.0, 0.1)
role_aggr = st.sidebar.slider("Agresividad / pressing (rol)", -2.0, 2.0, 0.3, 0.1)

x_ideal_role = [role_pos, role_vert, role_off, role_def, role_aggr]

# -------- Sidebar: estilo deseado del equipo --------
st.sidebar.header("3. Estilo deseado del equipo (x_coach)")

coach_pos = st.sidebar.slider("Posesión / elaboración (equipo)", -2.0, 2.0, 0.5, 0.1)
coach_vert = st.sidebar.slider("Verticalidad (equipo)", -2.0, 2.0, 0.5, 0.1)
coach_off = st.sidebar.slider("Ataque (equipo)", -2.0, 2.0, 0.7, 0.1)
coach_def = st.sidebar.slider("Defensa (equipo)", -2.0, 2.0, 0.3, 0.1)
coach_aggr = st.sidebar.slider("Agresividad / pressing (equipo)", -2.0, 2.0, 0.4, 0.1)

x_coach = [coach_pos, coach_vert, coach_off, coach_def, coach_aggr]

lam_coach = st.sidebar.slider("λ (peso preferencias del entrenador)", 0.0, 1.0, 0.5, 0.05)

# -------- Sidebar: pesos del score --------
st.sidebar.header("4. Pesos del score")

alpha = st.sidebar.slider("α · style_fit", 0.0, 1.0, 0.4, 0.05)
beta  = st.sidebar.slider("β · needs_fit", 0.0, 1.0, 0.3, 0.05)
gamma = st.sidebar.slider("γ · edad/potencial", 0.0, 1.0, 0.2, 0.05)
delta = st.sidebar.slider("δ · disciplina", 0.0, 1.0, 0.1, 0.05)

# Normalizar para que sumen 1 (si la suma > 0)
suma = alpha + beta + gamma + delta
if suma > 0:
    alpha, beta, gamma, delta = [w / suma for w in [alpha, beta, gamma, delta]]

st.sidebar.markdown(
    f"Pesos normalizados: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, δ={delta:.2f}"
)

# -------- Botón de ejecución --------
run_button = st.sidebar.button("🔍 Recomendar jugadores")

# ============================================================
# 8. Contenido principal
# ============================================================

if not run_button:
    st.info("Ajusta los parámetros en el panel lateral y pulsa **'🔍 Recomendar jugadores'**.")
else:
    try:
        recs, team_row, x_team_real, x_team_target = recommend_players_for_team(
            engine=engine,
            team_name=team_name,
            season=season,
            x_ideal_role=x_ideal_role,
            leagues=None if not solo_big5 else BIG5_LEAGUES,
            leagues_big5=solo_big5,
            min_minutes=min_minutes,
            role=role_param,
            x_coach=x_coach,
            lam_coach=lam_coach,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            top_n=top_n,
        )

        # ----- Mostrar estilo del equipo -----
        st.subheader(f"Perfil de estilo del equipo: {team_name} ({season})")

        dims_names = ["Posesión", "Verticalidad", "Ataque", "Defensa", "Agresividad/pressing"]
        df_style = pd.DataFrame(
            {
                "Dimension": dims_names,
                "Estilo_real": x_team_real,
                "Estilo_objetivo (real+coach)": x_team_target,
                "Rol_ideal": x_ideal_role,
            }
        )
        # Por ahora sin formato especial, para evitar el error
        
        st.dataframe(df_style, use_container_width=True)

        # ----- Tabla de recomendaciones -----
        st.subheader("Jugadores recomendados")

        st.markdown(
            """
Score = α·style_fit + β·needs_fit + γ·edad_potencial + δ·disciplina  
(Cada componente normalizado internamente)
"""
        )

        st.dataframe(recs, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al generar las recomendaciones: {e}")
