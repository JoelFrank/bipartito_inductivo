#!/bin/bash

# Script para ejecutar el entrenamiento completo de los modelos

echo "=== Entrenamiento de Modelos Bipartitos para Predicción de Enlaces ==="

# Configurar variables
DATASET_NAME="example-bipartite"
CONFIG_FILE="config/transductive_${DATASET_NAME}.cfg"

echo "Dataset: $DATASET_NAME"
echo "Archivo de configuración: $CONFIG_FILE"
echo ""

# Verificar que existe el dataset procesado
if [ ! -f "../data/processed/${DATASET_NAME}.pt" ]; then
    echo "ERROR: Dataset procesado no encontrado. Ejecuta primero el script de preparación de datos."
    echo "python data_preparation/create_my_dataset.py --csv ../data/raw/example_bipartite.csv --name $DATASET_NAME"
    exit 1
fi

echo "=== Entrenando T-BGRL (Triplet) ==="
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=triplet

echo ""
echo "=== Entrenando BGRL ==="
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=bgrl

echo ""
echo "=== Entrenando GBT (Graph Barlow Twins) ==="
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=gbt

echo ""
echo "=== Entrenando CCA-SSG ==="
python train_nc.py --flagfile="$CONFIG_FILE" --base_model=cca

echo ""
echo "=== Entrenando GRACE (baseline) ==="
python train_grace.py --flagfile="$CONFIG_FILE"

echo ""
echo "=== Entrenando E2E (baseline) ==="
python train_e2e.py --flagfile="$CONFIG_FILE"

echo ""
echo "=== Entrenando Margin (baseline) ==="
python train_margin.py --flagfile="$CONFIG_FILE"

echo ""
echo "=== Entrenamiento completo finalizado ==="
echo "Revisa los resultados en la carpeta '../runs/'"
