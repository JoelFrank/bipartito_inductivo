import json
import matplotlib.pyplot as plt
import pandas as pd

# Leer las métricas del archivo JSONL
metrics = []
with open('logs/bipartite-ncl-link-prediction/run_20250818_203340/metrics.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))

print("=== ANÁLISIS COMPARATIVO DE RESULTADOS ===")
print("Entrenamiento con 200 épocas vs entrenamiento anterior (100 épocas)")

# Extraer métricas finales
final_metrics = metrics[-1]

print(f"\n🔥 MEJORAS SIGNIFICATIVAS LOGRADAS:")
print(f"📈 AUC-ROC Test: 0.5738 → {final_metrics['final_test_auc_mean']:.4f} (+{final_metrics['final_test_auc_mean']-0.5738:.4f})")
print(f"📈 Hits@10:     0.85%  → {final_metrics['final_test_hits@10_mean']*100:.2f}% (+{(final_metrics['final_test_hits@10_mean']-0.0085)*100:.2f}%)")
print(f"📈 Hits@50:     3.81%  → {final_metrics['final_test_hits@50_mean']*100:.2f}% (+{(final_metrics['final_test_hits@50_mean']-0.0381)*100:.2f}%)")
print(f"📈 Hits@100:    7.51%  → {final_metrics['final_test_hits@100_mean']*100:.2f}% (+{(final_metrics['final_test_hits@100_mean']-0.0751)*100:.2f}%)")

print(f"\n=== MÉTRICAS DETALLADAS (200 ÉPOCAS) ===")
print(f"🎯 AUC-ROC Test:     {final_metrics['final_test_auc_mean']:.4f}")
print(f"🎯 Average Precision: {final_metrics['final_test_ap_mean']:.4f}")
print(f"🎯 Precision:        {final_metrics['final_test_precision_mean']:.4f}")
print(f"🎯 Recall:           {final_metrics['final_test_recall_mean']:.4f}")
print(f"🎯 F1-Score:         {final_metrics['final_test_f1_mean']:.4f}")

print(f"\n=== MÉTRICAS DE RANKING (Hits@K) ===")
print(f"🏆 Hits@10:  {final_metrics['final_test_hits@10_mean']*100:.2f}%")
print(f"🏆 Hits@50:  {final_metrics['final_test_hits@50_mean']*100:.2f}%")
print(f"🏆 Hits@100: {final_metrics['final_test_hits@100_mean']*100:.2f}%")

print(f"\n=== ANÁLISIS DE VALIDACIÓN ===")
print(f"📊 AUC-ROC Val:      {final_metrics['final_val_auc_mean']:.4f}")
print(f"📊 Hits@10 Val:     {final_metrics['final_val_hits@10_mean']*100:.2f}%")
print(f"📊 Hits@100 Val:    {final_metrics['final_val_hits@100_mean']*100:.2f}%")

# Analizar la evolución del loss durante entrenamiento
train_losses = []
epochs = []
for metric in metrics:
    if 'tbgrl_train_loss' in metric:
        train_losses.append(metric['tbgrl_train_loss'])
        epochs.append(metric['epoch'])

print(f"\n=== EVOLUCIÓN DEL TRAINING LOSS ===")
print(f"🔸 Loss inicial (Época 1):   {train_losses[0]:.4f}")
print(f"🔸 Loss final (Época 200):   {train_losses[-1]:.4f}")
print(f"🔸 Loss mínimo alcanzado:    {min(train_losses):.4f} (Época {epochs[train_losses.index(min(train_losses))]})")
print(f"🔸 Diferencia total:         {train_losses[-1] - train_losses[0]:.4f}")

# Identificar fases del entrenamiento
print(f"\n=== FASES DEL ENTRENAMIENTO ===")
# Fase 1: Épocas 1-50
phase1_avg = sum(train_losses[:50]) / 50
print(f"🔸 Fase 1 (1-50):    Promedio = {phase1_avg:.4f}")

# Fase 2: Épocas 51-100  
if len(train_losses) >= 100:
    phase2_avg = sum(train_losses[50:100]) / 50
    print(f"🔸 Fase 2 (51-100):  Promedio = {phase2_avg:.4f}")

# Fase 3: Épocas 101-150
if len(train_losses) >= 150:
    phase3_avg = sum(train_losses[100:150]) / 50
    print(f"🔸 Fase 3 (101-150): Promedio = {phase3_avg:.4f}")

# Fase 4: Épocas 151-200
if len(train_losses) >= 200:
    phase4_avg = sum(train_losses[150:200]) / 50
    print(f"🔸 Fase 4 (151-200): Promedio = {phase4_avg:.4f}")

print(f"\n=== DIAGNÓSTICO DE CONVERGENCIA ===")
last_20_losses = train_losses[-20:]
loss_std = pd.Series(last_20_losses).std()
print(f"🔸 Desviación estándar últimas 20 épocas: {loss_std:.4f}")
if loss_std < 0.01:
    print("✅ CONVERGENCIA: El modelo ha convergido (baja variabilidad)")
elif loss_std < 0.02:
    print("⚡ CONVERGIENDO: El modelo está cerca de converger")
else:
    print("🔄 ENTRENANDO: El modelo aún está aprendiendo")

print(f"\n=== COMPARACIÓN GENERAL ===")
improvement_auc = final_metrics['final_test_auc_mean'] - 0.5738
improvement_hits10 = final_metrics['final_test_hits@10_mean'] - 0.0085
improvement_hits100 = final_metrics['final_test_hits@100_mean'] - 0.0751

print(f"🚀 MEJORA EN AUC:     {improvement_auc:+.4f} ({improvement_auc/0.5738*100:+.1f}%)")
print(f"🚀 MEJORA EN HITS@10: {improvement_hits10:+.4f} ({improvement_hits10/0.0085*100:+.1f}%)")
print(f"🚀 MEJORA EN HITS@100:{improvement_hits100:+.4f} ({improvement_hits100/0.0751*100:+.1f}%)")

if improvement_auc > 0.01:
    print("🎉 EXCELENTE: Mejora significativa en AUC")
elif improvement_auc > 0.005:
    print("👍 BUENO: Mejora notable en AUC")
else:
    print("📊 LEVE: Mejora marginal en AUC")

print(f"\n=== PRÓXIMOS PASOS RECOMENDADOS ===")
if final_metrics['final_test_auc_mean'] > 0.65:
    print("✨ Excelente rendimiento! Considera:")
    print("  - Probar learning rates más bajos (0.001)")
    print("  - Aumentar dimensiones de embedding (256)")
    print("  - Evaluar en otros datasets")
elif final_metrics['final_test_auc_mean'] > 0.6:
    print("👍 Buen rendimiento! Próximas optimizaciones:")
    print("  - Aumentar a 300 épocas")
    print("  - Probar arquitecturas más complejas (GraphSAGE)")
    print("  - Ajuste fino de hiperparámetros")
else:
    print("📈 Rendimiento moderado. Estrategias:")
    print("  - Verificar calidad de datos")
    print("  - Probar diferentes learning rates")
    print("  - Considerar feature engineering")
