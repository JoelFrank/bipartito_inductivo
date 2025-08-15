# Guía de Instalación y Uso - Non-Contrastive Link Prediction Bipartite

## 📋 Requisitos Previos

### Sistema
- Python 3.8 o superior
- CUDA (opcional, para aceleración GPU)
- Git

### Librerías principales
- PyTorch
- PyTorch Geometric
- pandas, scikit-learn, tqdm

## 🚀 Instalación Paso a Paso

### 1. Clonar o Descargar el Proyecto
Si tienes Git:
```bash
git clone <repository-url>
cd non-contrastive-link-prediction-bipartite
```

O simplemente copia todos los archivos a una carpeta local.

### 2. Crear Entorno Virtual (Recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

**Nota**: Si tienes problemas con PyTorch Geometric, instálalo manualmente:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

## 📊 Preparación de Datos

### Opción 1: Usar Datos de Ejemplo
```bash
# Crear datos de ejemplo
python create_example_data.py

# Procesar los datos
python data_preparation/create_my_dataset.py --csv data/raw/example_bipartite.csv --name example-bipartite
```

### Opción 2: Usar Tus Propios Datos
1. Prepara un archivo CSV con dos columnas (ej: `user_id`, `item_id`)
2. Colócalo en `data/raw/`
3. Procésalo:
```bash
python data_preparation/create_my_dataset.py --csv data/raw/tu_archivo.csv --name mi-dataset
```

## 🏃‍♂️ Ejecutar Entrenamiento

### Opción 1: Experimento Individual
```bash
cd src
python train_nc.py --flagfile="config/transductive_my_dataset.cfg" --base_model=triplet
```

### Opción 2: Todos los Modelos (Windows)
```powershell
cd src
..\run_all_experiments.ps1
```

### Opción 3: Todos los Modelos (Linux/Mac)
```bash
cd src
bash ../run_all_experiments.sh
```

## 📈 Modelos Disponibles

### Modelos No Contrastivos (train_nc.py)
- **T-BGRL** (`--base_model=triplet`): Versión con pérdida triplete
- **BGRL** (`--base_model=bgrl`): Bootstrap Your Own Latent
- **GBT** (`--base_model=gbt`): Graph Barlow Twins
- **CCA-SSG** (`--base_model=cca`): Canonical Correlation Analysis

### Modelos de Referencia
- **GRACE** (`train_grace.py`): Graph Contrastive Learning
- **E2E** (`train_e2e.py`): End-to-End Training
- **Margin** (`train_margin.py`): Margin-based Loss

## ⚙️ Configuración

### Archivo de Configuración Principal
Edita `src/config/transductive_my_dataset.cfg`:

```ini
# Dataset
--dataset=mi-dataset
--dataset_dir=../data

# Entrenamiento
--lr=0.008
--epochs=5000
--base_model=triplet

# Arquitectura
--graph_encoder_layer_dims=128
--graph_encoder_layer_dims=128
```

### Parámetros Importantes
- `--dataset`: Nombre del dataset (sin extensión .pt)
- `--lr`: Tasa de aprendizaje
- `--epochs`: Número de épocas
- `--base_model`: Modelo a usar (triplet, bgrl, gbt, cca)
- `--graph_encoder_layer_dims`: Dimensiones de las capas del encoder

## 📊 Resultados

### Ubicación
Los resultados se guardan en:
```
runs/
├── Bipartite_triplet_mi-dataset_lr0.008_<timestamp>/
│   ├── train_nc_log.log
│   ├── agg_results.json
│   └── config.cfg
logs/
├── <project-name>/
│   └── run_<timestamp>/
│       ├── metrics.jsonl
│       ├── summary.json
│       └── config.json
```

### Métricas Evaluadas
- **AUC**: Área bajo la curva ROC
- **AP**: Average Precision
- **Hits@K**: Hits a 10, 50 y 100

### Visualización
Los resultados se registran en archivos JSON locales para análisis posterior. Puedes usar herramientas como pandas para visualizar las métricas.

## 🐛 Solución de Problemas Comunes

### Error: "Dataset not found"
```bash
# Verifica que el archivo existe
ls data/processed/mi-dataset.pt

# Si no existe, procesa los datos nuevamente
python data_preparation/create_my_dataset.py --csv data/raw/mi_archivo.csv --name mi-dataset
```

### Error: "CUDA out of memory"
- Reduce el tamaño del batch o las dimensiones del modelo
- Usa CPU: elimina `.to(device)` en el código

### Error: "Module not found"
```bash
# Instala dependencias faltantes
pip install -r requirements.txt

# O específicamente:
pip install torch torch-geometric pandas scikit-learn
```

## 📝 Personalización

### Agregar Nuevo Dataset
1. Coloca tu CSV en `data/raw/`
2. Ejecuta el script de preparación
3. Crea nuevo archivo de configuración en `src/config/`
4. Actualiza el parámetro `--dataset`

### Modificar Hiperparámetros
- Edita los archivos `.cfg` en `src/config/`
- O pasa parámetros directamente: `--lr=0.01 --epochs=1000`

### Agregar Nuevas Métricas
- Modifica `src/lib/eval.py`
- Agrega las métricas en la función `eval_all()`

## 📞 Soporte

Si encuentras problemas:
1. Revisa los logs en `runs/*/train_*_log.log`
2. Verifica que todas las dependencias estén instaladas
3. Consulta la documentación de PyTorch Geometric
4. Revisa los issues del repositorio original

## 🎯 Ejemplo Completo de Uso

```bash
# 1. Activar entorno
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Crear datos de ejemplo
python create_example_data.py

# 3. Procesar datos
python data_preparation/create_my_dataset.py --csv data/raw/example_bipartite.csv --name example-bipartite

# 4. Entrenar modelo
cd src
python train_nc.py --flagfile="config/transductive_my_dataset.cfg" --base_model=triplet --dataset=example-bipartite

# 5. Ver resultados
# Revisa runs/ y logs/ para los resultados detallados
```

¡Listo para empezar! 🚀
