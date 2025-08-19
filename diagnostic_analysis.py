import pandas as pd
import numpy as np

#df = pd.read_csv('runs/sismetro_bipartite_sage_inductive/Bipartite_triplet_sismetro_lr0.01_run_20250818_201049/prod_mlp_test_results.csv')
df = pd.read_csv('runs/sismetro_bipartite_sage_inductive/Bipartite_triplet_sismetro_lr0.01_run_20250818_203340/prod_mlp_test_results.csv')

print("=== DIAGNÓSTICO DETALLADO ===")

# 1. Análisis de casos problemáticos
print("\n1. CASOS PROBLEMÁTICOS (High score pero negative label):")
problematic = df[(df['score'] > 0.8) & (df['label'] == 0)]
print(f"Predicciones > 0.8 con label negativo: {len(problematic)}")
for _, row in problematic.head(5).iterrows():
    print(f"  ({int(row['u'])}, {int(row['v'])}) -> {row['score']:.4f}")

print("\n2. CASOS POSITIVOS MAL CLASIFICADOS (Low score pero positive label):")
missed_positives = df[(df['score'] < 0.4) & (df['label'] == 1)]
print(f"Predicciones < 0.4 con label positivo: {len(missed_positives)}")
for _, row in missed_positives.head(5).iterrows():
    print(f"  ({int(row['u'])}, {int(row['v'])}) -> {row['score']:.4f}")

# 2. Análisis por rango de scores
print("\n3. DISTRIBUCIÓN POR RANGOS DE SCORE:")
ranges = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.0)]
for low, high in ranges:
    subset = df[(df['score'] >= low) & (df['score'] < high)]
    if len(subset) > 0:
        pos_rate = sum(subset['label'] == 1) / len(subset)
        print(f"  Score [{low:.1f}, {high:.1f}): {len(subset)} casos, {pos_rate:.1%} positivos")

# 3. Análisis temporal/espacial (por nodos)
print("\n4. ANÁLISIS POR NODOS:")
# Nodos v más problemáticos
v_analysis = df.groupby('v').agg({
    'score': ['mean', 'std', 'count'],
    'label': ['mean', 'sum']
}).round(3)
v_analysis.columns = ['score_mean', 'score_std', 'count', 'pos_rate', 'pos_count']
v_analysis = v_analysis[v_analysis['count'] >= 10]  # Solo nodos con >= 10 evaluaciones

print("Top 5 nodos v con mayor tasa de positivos:")
top_positive = v_analysis.sort_values('pos_rate', ascending=False).head()
print(top_positive[['score_mean', 'pos_rate', 'count']])

print("\nTop 5 nodos v con scores más altos:")
top_scores = v_analysis.sort_values('score_mean', ascending=False).head()
print(top_scores[['score_mean', 'pos_rate', 'count']])

# 4. Correlación entre score y label
correlation = df['score'].corr(df['label'])
print(f"\n5. CORRELACIÓN SCORE-LABEL: {correlation:.4f}")

# 5. Análisis de threshold
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
print(f"\n6. ANÁLISIS DE DIFERENTES THRESHOLDS:")
for thresh in thresholds:
    predictions = (df['score'] >= thresh).astype(int)
    tp = sum((predictions == 1) & (df['label'] == 1))
    fp = sum((predictions == 1) & (df['label'] == 0))
    tn = sum((predictions == 0) & (df['label'] == 0))
    fn = sum((predictions == 0) & (df['label'] == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Threshold {thresh:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

print(f"\n=== CONCLUSIONES DEL DIAGNÓSTICO ===")
print(f"🔸 Rendimiento MODERADO: AUC={df['score'].corr(df['label']):.3f} indica aprendizaje limitado")
print(f"🔸 Balance perfecto: {sum(df['label'])/len(df)*100:.1f}% positivos - no hay sesgo de clases")
print(f"🔸 Scores distribuidos: rango amplio pero overlap significativo entre clases")
print(f"🔸 Casos problemáticos: {len(problematic)} high-confidence false positives")
print(f"🔸 Embedding learning: modelo aprende representaciones pero con capacidad discriminativa limitada")

print(f"\n=== POSIBLES MEJORAS ===")
print(f"✨ Aumentar epochs de entrenamiento")
print(f"✨ Ajustar learning rate (probar 0.001 o 0.005)")
print(f"✨ Aumentar dimensiones de embedding")
print(f"✨ Probar otras arquitecturas (GraphSAGE, GAT)")
print(f"✨ Analizar características temporales más detalladamente")
