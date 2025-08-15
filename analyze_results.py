#!/usr/bin/env python3
"""
Script para analizar y visualizar los resultados del entrenamiento
Reemplaza la funcionalidad de visualización de wandb
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics_from_jsonl(filepath):
    """Cargar métricas desde archivo JSONL"""
    metrics = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                metrics.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filepath}")
    return metrics

def load_all_results(base_dir="runs"):
    """Cargar todos los resultados de experimentos"""
    results = {}
    
    # Buscar archivos de resultados agregados
    for result_file in glob.glob(f"{base_dir}/**/agg_results.json", recursive=True):
        experiment_name = os.path.basename(os.path.dirname(result_file))
        try:
            with open(result_file, 'r') as f:
                results[experiment_name] = json.load(f)
        except Exception as e:
            print(f"Error cargando {result_file}: {e}")
    
    return results

def load_training_metrics(logs_dir="logs"):
    """Cargar métricas de entrenamiento desde logs"""
    training_data = {}
    
    if not os.path.exists(logs_dir):
        print(f"Directorio de logs no encontrado: {logs_dir}")
        return training_data
    
    for project_dir in os.listdir(logs_dir):
        project_path = os.path.join(logs_dir, project_dir)
        if os.path.isdir(project_path):
            for run_dir in os.listdir(project_path):
                run_path = os.path.join(project_path, run_dir)
                metrics_file = os.path.join(run_path, "metrics.jsonl")
                if os.path.exists(metrics_file):
                    run_name = f"{project_dir}_{run_dir}"
                    training_data[run_name] = load_metrics_from_jsonl(metrics_file)
    
    return training_data

def plot_training_curves(training_data, output_dir="analysis"):
    """Crear gráficos de curvas de entrenamiento"""
    os.makedirs(output_dir, exist_ok=True)
    
    for run_name, metrics in training_data.items():
        if not metrics:
            continue
        
        df = pd.DataFrame(metrics)
        
        # Buscar columnas de loss
        loss_columns = [col for col in df.columns if 'loss' in col.lower()]
        
        if loss_columns:
            plt.figure(figsize=(10, 6))
            for loss_col in loss_columns:
                if loss_col in df.columns:
                    plt.plot(df['timestamp'], df[loss_col], label=loss_col)
            
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Loss')
            plt.title(f'Curvas de Pérdida - {run_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{run_name}_loss.png'))
            plt.close()

def create_results_comparison(results, output_dir="analysis"):
    """Crear tabla comparativa de resultados"""
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_data = []
    
    for experiment, metrics in results.items():
        row = {'experiment': experiment}
        
        # Extraer métricas principales
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Guardar como CSV
        df.to_csv(os.path.join(output_dir, 'results_comparison.csv'), index=False)
        
        # Crear gráfico comparativo si hay métricas de test
        test_metrics = [col for col in df.columns if 'test' in col.lower() and 'mean' in col.lower()]
        
        if test_metrics:
            plt.figure(figsize=(12, 8))
            df_plot = df.set_index('experiment')[test_metrics]
            df_plot.plot(kind='bar', ax=plt.gca())
            plt.title('Comparación de Métricas de Test')
            plt.ylabel('Valor de Métrica')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'test_metrics_comparison.png'))
            plt.close()
        
        print(f"Tabla de comparación guardada en: {output_dir}/results_comparison.csv")
        return df
    
    return None

def generate_report(output_dir="analysis"):
    """Generar reporte completo de resultados"""
    print("=== Análisis de Resultados ===")
    
    # Cargar datos
    results = load_all_results()
    training_data = load_training_metrics()
    
    print(f"Experimentos encontrados: {len(results)}")
    print(f"Runs de entrenamiento: {len(training_data)}")
    
    if not results and not training_data:
        print("No se encontraron resultados para analizar.")
        return
    
    # Crear visualizaciones
    if training_data:
        print("Generando curvas de entrenamiento...")
        plot_training_curves(training_data, output_dir)
    
    if results:
        print("Creando comparación de resultados...")
        df_comparison = create_results_comparison(results, output_dir)
        
        if df_comparison is not None:
            print("\n=== Resumen de Resultados ===")
            print(df_comparison.to_string(index=False))
    
    print(f"\nAnálisis completo guardado en: {output_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizar resultados de experimentos")
    parser.add_argument('--output', default='analysis', help='Directorio de salida para análisis')
    parser.add_argument('--runs-dir', default='runs', help='Directorio de resultados')
    parser.add_argument('--logs-dir', default='logs', help='Directorio de logs')
    
    args = parser.parse_args()
    
    # Cambiar directorios de trabajo si se especifican
    if args.runs_dir != 'runs':
        global load_all_results
        load_all_results = lambda: load_all_results(args.runs_dir)
    
    if args.logs_dir != 'logs':
        global load_training_metrics
        load_training_metrics = lambda: load_training_metrics(args.logs_dir)
    
    generate_report(args.output)
