# Recomendación de Jugadores de Fútbol basada en CRISP-DM

Este repositorio contiene un proyecto de **recomendación de jugadores de fútbol** basado en la metodología **CRISP-DM** y en un flujo moderno de **data warehouse por capas (Bronze / Silver / Gold)**.  

El objetivo es construir un sistema que permita:
- Explorar y comparar jugadores.
- Definir **métricas personalizadas** de rendimiento y estilo de juego.
- Entrenar **modelos de Machine Learning** para apoyar la recomendación de jugadores según distintas necesidades (fichajes, scouting, reemplazos, etc.).

## 1. Metodología: CRISP-DM aplicada al proyecto

Este proyecto sigue la metodología **CRISP-DM (Cross Industry Standard Process for Data Mining)** adaptada al contexto de análisis y recomendación de jugadores.

### 1.1. Business Understanding (Entendimiento del negocio)

- **Problema principal:** apoyar el proceso de scouting y fichajes mediante recomendaciones de jugadores basadas en datos objetivos.
- **Preguntas clave:**
  - ¿Qué jugadores son similares a un jugador referencia (por rendimiento / perfil táctico)?
  - ¿Qué jugadores podrían ser buenos reemplazos si se va una pieza clave?
  - ¿Qué jugadores tienen métricas de rendimiento acordes a cierto estilo de juego (posesión, presión alta, transiciones rápidas, etc.)?
- **Usuarios objetivo:**
  - Analistas de datos de clubes.
  - Scouters / ojeadores.
  - Equipos de data science interesados en fútbol.

### 1.2. Data Understanding (Entendimiento de los datos)

- **Fuente principal de datos:** [FBref](https://fbref.com/)  
- **Formato original:** archivos CSV descargados manualmente (o mediante scripts) con estadísticas avanzadas de jugadores.
- **Competencias y ligas:** (ejemplo, ajustar según el proyecto real)
  - LaLiga, Premier League, etc.
  - Temporadas 20XX–20YY.
- **Tipos de datos:**
  - Datos básicos: nombre, equipo, liga, posición, edad, minutos jugados.
  - Métricas ofensivas: goles, xG, asistencias, xA, tiros, acciones de creación de tiro (SCA), etc.
  - Métricas defensivas: intercepciones, entradas, presiones, duelos, etc.
  - Métricas de pase: pases completados, progresivos, claves, etc.

### 1.3. Data Preparation (Preparación de los datos)

Se sigue una arquitectura de **data warehouse por capas**:

#### Capa Bronze (Raw / Staging)

- CSVs descargados directamente de FBref.
- Estructura fiel a la fuente original (mínimas transformaciones).
- Almacenamiento en un **data warehouse** (por ejemplo: MySQL, SQL Server, Spark, etc.).
- Tareas típicas:
  - Carga de CSV → tablas “raw”.
  - Registro de metadatos (temporada, liga, fecha de descarga).

#### Capa Silver (Limpieza y estandarización)

- Normalización de:
  - Nombres de jugadores.
  - Nombres de equipos.
  - Posiciones (ej. convertir “FW, ST” → “Delantero Centro”).
- Manejo de valores faltantes:
  - Imputación.
  - Filtrado de jugadores con pocos minutos.
- Conversión de tipos:
  - Campos numéricos.
  - Fechas.
- Creación de tablas unificadas por temporada / liga / competencia.


### 1.4. Modeling (Modelado)

Se implementan distintos enfoques de **recomendación de jugadores**:

#### 1.4.1. Métricas y sistema de recomendación “clásico”

- Normalización de estadísticas por posición.
- Cálculo de **distancias / similitudes** entre jugadores, por ejemplo:
  - Distancia euclidiana.
  - Cosine similarity.
- Recomendación de jugadores:
  - “Jugadores más parecidos a X”.
  - “Top N jugadores con mejor combinación de métricas ofensivas/defensivas”.

#### 1.4.2. Modelos de Machine Learning

Ejemplos de enfoques, ajustables según el repositorio:

- **Clustering** (no supervisado):
  - K-Means, GMM, etc. para segmentar jugadores en “perfiles” o “roles”.
- **Modelos supervisados**:
  - Regresión (predecir contribuciones ofensivas, goles + asistencias, contribución esperada, rating, etc.).
  - Clasificación (ej. etiquetar jugadores en clusters o categorías de rendimiento alto/medio/bajo).
- **Modelos de ranking / recomendación**:
  - Score final por jugador basado en:
    - Métricas normalizadas.
    - Pesos definidos por el usuario (ej. ofensivo 50%, pase 30%, defensa 20%).
    - Posibles modelos de aprendizaje a ranking (si se cuenta con datos de “target”).

### 1.5. Evaluation (Evaluación)

- **Evaluación de modelos de ML:**
  - Métricas de clasificación: accuracy, F1, recall, precision (si aplica).
  - Métricas de regresión: RMSE, MAE, R².
  - Evaluación de clustering: silhouette score, análisis cualitativo de grupos.
- **Evaluación del sistema de recomendación:**
  - Coherencia futbolística de las recomendaciones (validación cualitativa).
  - Comparación de jugadores recomendados vs. conocimiento experto.
  - Ejemplos de “casos de uso”:
    - Encontrar reemplazo de un jugador vendido.
    - Buscar perfiles específicos (lateral ofensivo, mediocentro de posesión, etc.).

### 1.6. Deployment (Despliegue)

Dependiendo del alcance del proyecto, se pueden implementar:

- Scripts en Python para generar recomendaciones a partir de un jugador objetivo.
- API sencilla (Flask/FastAPI) para consumir el modelo.
- Dashboard (ej. Streamlit / Dash) para:
  - Buscar un jugador.
  - Ver sus métricas.
  - Obtener recomendaciones similares.

---

## 2. Arquitectura del proyecto

### 2.1. Tecnologías usadas

- **Lenguaje:** Python
- **Librerías principales:**
  - `pandas`, `numpy` para manipulación de datos.
  - `scikit-learn` para modelos de ML y métricas.
  - `sqlalchemy` / conectores específicos para el data warehouse.
- **Data Warehouse:** (rellenar según tu entorno: MySQL)

### 2.2. Estructura del repositorio (propuesta)

```text
.
├── data/
│   ├── raw/                 # CSV originales de FBref (Bronze)
│   ├── interim/             # Datos limpios intermedios (Silver)
│   └── processed/           # Tablas listas para modelado (Gold)
├── notebooks/
│   ├── 01_entendimiento_datos.ipynb
│   ├── 02_preparacion_bronze_silver.ipynb
│   ├── 03_creacion_metricas_gold.ipynb
│   ├── 04_modelos_recomendacion.ipynb
│   └── 05_evaluacion_modelos.ipynb
├── src/
│   ├── data/
│   │   ├── load_fbref.py            # Carga de CSV a Bronze
│   │   ├── transform_silver.py      # Limpieza y normalización
│   │   └── build_gold_features.py   # Features finales
│   ├── models/
│   │   ├── similarity.py            # Funciones de recomendación por similitud
│   │   ├── clustering.py            # Modelos de clustering
│   │   └── supervised.py            # Modelos supervisados
│   └── utils/
│       └── config.py                # Configuración de rutas, conexión a DWH, etc.
├── README.md
└── requirements.txt

