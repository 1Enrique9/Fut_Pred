# ⚽ Fútbol Data 2024-2025 · Pipeline Bronze–Silver–Gold

Equipo: 
- Herrera Barron Fabia 
- Morales Flores Luis Enrique
- Villalon Pineda Luis Enrique

Proyecto de **ingeniería de datos y recomendación de jugadores** usando estadísticas de las 5 grandes ligas europeas (Alemania, España, Francia, Inglaterra e Italia).


La idea general del proyecto es:

1. Recolectar datos crudos de equipos y jugadores.  
2. Limpiarlos y normalizarlos adecuadamente.  
3. Cargarlos a una base de datos relacional bajo un esquema en capas **Bronze → Silver → Gold**.  
4. Construir un **sistema de recomendación de jugadores** basado en estilo del equipo + preferencias del entrenador.  
5. Visualizar todo mediante un **dashboard en Streamlit**.

---

## 📂 Estructura del repositorio

```text
.
├── Data_equipos/                  # Datos crudos de equipos por liga
│   └── 2024-2025/
│       ├── Alemania/
│       ├── España/
│       ├── Francia/
│       ├── Inglaterra/
│       └── Italia/
│           └── *.csv             # Un CSV por equipo
│
├── Datos_jugadores/               # Datos crudos de jugadores
│   └── 2024-2025/
│       ├── Alemania/
│       ├── España/
│       ├── Francia/
│       ├── Inglaterra/
│       └── Italia/
│           └── <Equipo>/
│               ├── a1.csv        # Particiones del equipo (a1, a2, a3…)
│               ├── a2.csv
│               ├── a3.csv
│               └── ...
│           └── Combinados/
│               └── Arsenal.csv   # Unión final de los archivos por equipo
│
├── Bronze.ipynb                   # Capa Bronze: ingesta de datos en MySQL
├── Silver.ipynb                   # Capa Silver: limpieza + features avanzadas
├── Gold_b.ipynb                   # Capa Gold: sistema de recomendación
├── Union_jugadores.ipynb          # Unión de particiones (a1,a2,a3…) por equipo
├── Correccion_de_datasets.ipynb   # Corrección/limpieza de Excel → CSV
├── dash.py                        # Dashboard en Streamlit
└── README.md
```

> 🔎 **Nota importante:** En `Datos_jugadores/…/<Equipo>/` cada equipo contiene varios archivos divididos (a1, a2, a3…).  
> `Union_jugadores.ipynb` concatena todos y genera un CSV único en `Combinados/`.

---

## 🧱 Pipeline completo de datos

### **0. Corrección de datasets** (`Correccion_de_datasets.ipynb`)
- Convierte Excel con encabezados multinivel a CSV limpios.
- Elimina columnas `Unnamed`.
- Unifica los nombres de columnas.
- Exporta todos los archivos limpios para que Bronze pueda ingerirlos.

---

### **1. Unión de particiones por equipo** (`Union_jugadores.ipynb`)
- Cada equipo tiene varios archivos: `a1.csv`, `a2.csv`, `a3.csv`, etc.
- Este notebook:
  - Lee todos los sub-archivos del equipo.  
  - Los concatena en un solo DataFrame.  
  - Genera un archivo final en:

```text
Datos_jugadores/2024-2025/<Liga>/Combinados/<Equipo>.csv
```

---

### **2. Capa Bronze** (`Bronze.ipynb`)
- Carga directa de CSV → MySQL **sin transformar nada**, solo organizando.
- Tablas principales:
  - `bronze_teams`
  - `bronze_players`
- Agrega metadatos:
  - `league`, `season`, `team_name`, `file_source`, etc.
- Garantiza consistencia y evita duplicados.

---

### **3. Capa Silver** (`Silver.ipynb`)
Objetivo: limpiar, estandarizar y generar **features de calidad**.

Incluye:

- Limpieza profunda de nombres de columnas.
- Conversión a `snake_case`.
- Cálculo de estadísticas por 90':
  - goles, asistencias, tiros, pases progresivos, disputas, etc.
- Cálculo de **z-scores** por:
  - posición,
  - liga,
  - rol de jugador.
- Creación de índices avanzados:
  - `idx_finishing`,  
  - `idx_playmaking`,  
  - `idx_progression`,  
  - `idx_involvement`,  
  - `idx_defending`,  
  - `idx_discipline`, etc.

Genera las tablas:

- `silver_players`
- `silver_teams`

---

### **4. Capa Gold: Sistema de Recomendación** (`Gold_b.ipynb`)
Construye un recomendador basado en:

1. Estilo real del equipo (`get_team_profile`)  
2. Preferencias del entrenador (`x_coach`)  
3. Perfil ideal para el rol (delantero, mediocampo, defensa, etc.)  
4. Fórmulas de similitud ponderada para obtener el **top-N** de jugadores ideales.

Funciones clave:

- `get_team_profile(...)`
- `get_player_universe(...)`
- `build_team_target_vector(...)`
- `recommend_players_for_team(...)`

Ejemplo general:

```python
recs = recommend_players_for_team(
    engine=engine,
    team_name="Barcelona",
    season="2024-2025",
    x_ideal_role=[0.5, 1.0, 1.0, 0.0, 0.3],
    leagues_big5=True,
    min_minutes=600,
    role="delantero",
    x_coach=[0.6, 0.7, 0.9, 0.3, 0.4],
    lam_coach=0.5,
    alpha=0.4, beta=0.3, gamma=0.2, delta=0.1,
    top_n=30,
)
```

---

### **5. Dashboard en Streamlit** (`dash.py`)

Permite:

- Seleccionar equipo y temporada.  
- Elegir rol (delantero, mediocampista, defensa, portero).  
- Mover sliders para preferencias del entrenador.  
- Obtener recomendaciones en tiempo real.

Para ejecutar:

```bash
streamlit run dash.py
```

---

## 🚀 Cómo reproducir

1. Clonar el repositorio.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar credenciales MySQL.
4. Ejecutar **en este orden**:

```
0. Correccion_de_datasets.ipynb
1. Union_jugadores.ipynb
2. Bronze.ipynb
3. Silver.ipynb
4. Gold_b.ipynb
5. streamlit run dash.py
```

---

## 📌 Posibles mejoras

- Integrar datos de valor de mercado o salarios.
- Incluir más temporadas.
- Crear filtros avanzados (edad, minutos jugados, contrato).
- Exportar rankings automáticos en CSV/Excel.

---

