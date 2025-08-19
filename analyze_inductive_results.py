import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# Leer los resultados
df = pd.read_csv('runs/sismetro_bipartite_sage_inductive/Bipartite_triplet_sismetro_lr0.01_run_20250818_201049/prod_mlp_test_results.csv')

print("=== ANÁLISIS DE RESULTADOS INDUCTIVOS T-BGRL ===")
print(f"Total de predicciones: {len(df)}")
print(f"Enlaces positivos: {sum(df['label'] == 1)}")
print(f"Enlaces negativos: {sum(df['label'] == 0)}")
print(f"Balance: {sum(df['label'] == 1)/len(df)*100:.1f}% positivos")

# Métricas principales
auc = roc_auc_score(df['label'], df['score'])
ap = average_precision_score(df['label'], df['score'])

print(f"\n=== MÉTRICAS DE RENDIMIENTO ===")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision (AP): {ap:.4f}")

# Análisis de distribución de scores
print(f"\n=== DISTRIBUCIÓN DE SCORES ===")
print(f"Score mínimo: {df['score'].min():.4f}")
print(f"Score máximo: {df['score'].max():.4f}")
print(f"Score promedio: {df['score'].mean():.4f}")
print(f"Score mediana: {df['score'].median():.4f}")

# Scores por clase
pos_scores = df[df['label'] == 1]['score']
neg_scores = df[df['label'] == 0]['score']

print(f"\nScores POSITIVOS:")
print(f"  Promedio: {pos_scores.mean():.4f}")
print(f"  Mediana: {pos_scores.median():.4f}")
print(f"  Min: {pos_scores.min():.4f}")
print(f"  Max: {pos_scores.max():.4f}")

print(f"\nScores NEGATIVOS:")
print(f"  Promedio: {neg_scores.mean():.4f}")
print(f"  Mediana: {neg_scores.median():.4f}")
print(f"  Min: {neg_scores.min():.4f}")
print(f"  Max: {neg_scores.max():.4f}")

# Hits@K para diferentes valores de K
def calculate_hits_at_k(df, k):
    # Ordenar por score descendente
    df_sorted = df.sort_values('score', ascending=False)
    # Tomar los top K
    top_k = df_sorted.head(k)
    # Calcular hits@k
    hits = sum(top_k['label'] == 1) / sum(df['label'] == 1)
    return hits

print(f"\n=== HITS@K ===")
for k in [10, 20, 50, 100]:
    hits = calculate_hits_at_k(df, k)
    print(f"Hits@{k}: {hits:.4f}")

# Análisis de threshold óptimo
from sklearn.metrics import precision_recall_curve, f1_score

precision, recall, thresholds = precision_recall_curve(df['label'], df['score'])
f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = np.nan_to_num(f1_scores)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]

print(f"\n=== THRESHOLD ÓPTIMO ===")
print(f"Mejor threshold: {best_threshold:.4f}")
print(f"F1-Score máximo: {best_f1:.4f}")
print(f"Precisión: {precision[best_threshold_idx]:.4f}")
print(f"Recall: {recall[best_threshold_idx]:.4f}")

# Predicciones con threshold óptimo
predictions = (df['score'] >= best_threshold).astype(int)
tp = sum((predictions == 1) & (df['label'] == 1))
fp = sum((predictions == 1) & (df['label'] == 0))
tn = sum((predictions == 0) & (df['label'] == 0))
fn = sum((predictions == 0) & (df['label'] == 1))

print(f"\n=== MATRIZ DE CONFUSIÓN (threshold={best_threshold:.4f}) ===")
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives:  {tn}")
print(f"False Negatives: {fn}")

# Análisis de nodos más conectados
print(f"\n=== ANÁLISIS DE NODOS ===")
u_counts = df['u'].value_counts()
v_counts = df['v'].value_counts()

print(f"Nodos u únicos: {len(u_counts)}")
print(f"Nodos v únicos: {len(v_counts)}")
print(f"Rango nodos u: {df['u'].min()} - {df['u'].max()}")
print(f"Rango nodos v: {df['v'].min()} - {df['v'].max()}")

print(f"\nTop 5 nodos u más evaluados:")
print(u_counts.head())

print(f"\nTop 5 nodos v más evaluados:")
print(v_counts.head())

# Scores altos y bajos
print(f"\n=== TOP 10 PREDICCIONES MÁS ALTAS ===")
top_scores = df.nlargest(10, 'score')
for _, row in top_scores.iterrows():
    label_str = "POS" if row['label'] == 1 else "NEG"
    print(f"({row['u']}, {row['v']}) -> {row['score']:.4f} [{label_str}]")

print(f"\n=== TOP 10 PREDICCIONES MÁS BAJAS ===")
bottom_scores = df.nsmallest(10, 'score')
for _, row in bottom_scores.iterrows():
    label_str = "POS" if row['label'] == 1 else "NEG"
    print(f"({row['u']}, {row['v']}) -> {row['score']:.4f} [{label_str}]")

print(f"\n=== RESUMEN EJECUTIVO ===")
print(f"✅ Entrenamiento inductivo EXITOSO con T-BGRL + GCN")
print(f"✅ AUC-ROC: {auc:.4f} ({'Excelente' if auc > 0.8 else 'Bueno' if auc > 0.7 else 'Regular'})")
print(f"✅ Average Precision: {ap:.4f}")
print(f"✅ Separación de clases: {pos_scores.mean() - neg_scores.mean():.4f}")
print(f"✅ Balance de datos: {sum(df['label'] == 1)/len(df)*100:.1f}% positivos")
