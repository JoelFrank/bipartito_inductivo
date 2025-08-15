# Script PowerShell para ejecutar el entrenamiento completo de los modelos

Write-Host "=== Entrenamiento de Modelos Bipartitos para Predicción de Enlaces ===" -ForegroundColor Green

# Configurar variables
$DATASET_NAME = "example-bipartite"
$CONFIG_FILE = "config/transductive_$DATASET_NAME.cfg"

Write-Host "Dataset: $DATASET_NAME"
Write-Host "Archivo de configuración: $CONFIG_FILE"
Write-Host ""

# Verificar que existe el dataset procesado
$datasetPath = "../data/processed/$DATASET_NAME.pt"
if (-not (Test-Path $datasetPath)) {
    Write-Host "ERROR: Dataset procesado no encontrado. Ejecuta primero el script de preparación de datos." -ForegroundColor Red
    Write-Host "python data_preparation/create_my_dataset.py --csv ../data/raw/example_bipartite.csv --name $DATASET_NAME" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== Entrenando T-BGRL (Triplet) ===" -ForegroundColor Cyan
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=triplet

Write-Host ""
Write-Host "=== Entrenando BGRL ===" -ForegroundColor Cyan
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=bgrl

Write-Host ""
Write-Host "=== Entrenando GBT (Graph Barlow Twins) ===" -ForegroundColor Cyan
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=gbt

Write-Host ""
Write-Host "=== Entrenando CCA-SSG ===" -ForegroundColor Cyan
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=cca

Write-Host ""
Write-Host "=== Entrenando GRACE (baseline) ===" -ForegroundColor Cyan
python train_grace.py --flagfile="$CONFIG_FILE"

Write-Host ""
Write-Host "=== Entrenando E2E (baseline) ===" -ForegroundColor Cyan
python train_e2e.py --flagfile="$CONFIG_FILE"

Write-Host ""
Write-Host "=== Entrenando Margin (baseline) ===" -ForegroundColor Cyan
python train_margin.py --flagfile="$CONFIG_FILE"

Write-Host ""
Write-Host "=== Entrenamiento completo finalizado ===" -ForegroundColor Green
Write-Host "Revisa los resultados en la carpeta '../runs/'" -ForegroundColor Yellow
