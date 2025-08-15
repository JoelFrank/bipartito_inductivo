# Predicción de Enlaces con Aprendizaje No Contrastivo (Adaptación Bipartita)

Este proyecto adapta el trabajo original "Link Prediction with Non-Contrastive Learning" para operar sobre **grafos bipartitos, no ponderados y no dirigidos**. El objetivo es evaluar el rendimiento de modelos como T-BGRL, BGRL, GBT y CCA-SSG en tareas de predicción de enlaces en este tipo de grafos, comunes en sistemas de recomendación.

## Flujo de Trabajo

### 1. Preparación de Datos
Antes de entrenar, debes convertir tus datos crudos (ej. un archivo CSV con interacciones `usuario-item`) a un formato de grafo de PyTorch Geometric.

Usa el script `data_preparation/create_my_dataset.py`. Desde la raíz del proyecto, ejecuta:

```bash
python data_preparation/create_my_dataset.py --csv path/to/your/data.csv --name my-bipartite-dataset
```

Esto creará un archivo `data/processed/my-bipartite-dataset.pt`.

### 2. Configuración del Experimento

Crea un archivo de configuración en `src/config/`. Puedes basarte en `transductive_my_dataset.cfg`. Asegúrate de que el flag `--dataset` coincida con el nombre que le diste a tu grafo en el paso anterior.

### 3. Ejecución del Entrenamiento

Navega a la carpeta `src` y lanza el entrenamiento usando tu archivo de configuración:

```bash
cd src
python train_nc.py --flagfile="config/transductive_my-dataset.cfg"
```

Para los modelos de referencia:

```bash
python train_grace.py --flagfile="config/transductive_my-dataset.cfg"
python train_e2e.py --flagfile="config/transductive_my-dataset.cfg"
python train_margin.py --flagfile="config/transductive_my-dataset.cfg"
```

### 4. Revisión de Resultados

Los resultados, logs y configuraciones se guardarán en el directorio especificado por `--logdir` en tu archivo de configuración (por defecto, `src/runs/`). Cada ejecución tendrá una carpeta única que contiene:

- `train_log.log`: El log completo del proceso.
- `agg_results.json`: Los resultados finales de la evaluación.
- `config.cfg`: La configuración usada para la ejecución.
- `logs/`: Métricas detalladas de entrenamiento en formato JSON.

### 5. Análisis de Resultados

Para analizar y visualizar los resultados de todos los experimentos:

```bash
python analyze_results.py
```

Esto generará:
- Gráficos de curvas de pérdida durante el entrenamiento
- Tabla comparativa de métricas entre modelos
- Gráficos comparativos de rendimiento

## Estructura del Proyecto

```
non-contrastive-link-prediction-bipartite/
├── data_preparation/
│   └── create_my_dataset.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── config/
│   │   └── transductive_my_dataset.cfg
│   ├── lib/
│   │   ├── models/
│   │   └── training/
│   ├── train_nc.py
│   ├── train_grace.py
│   ├── train_e2e.py
│   └── train_margin.py
├── README.md
└── requirements.txt
```

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Prepara tu dataset usando el script de preparación de datos.

3. Configura tu experimento y ejecuta el entrenamiento.
