# Dataset Sismetro - Split Inductivo Temporal

## ✅ Split Inductivo Implementado Exitosamente

### 📊 Configuración del Split Temporal:

- **80% TRAIN (Pasado)**: Datos históricos para entrenamiento
- **10% VAL (Presente)**: Datos recientes para validación 
- **10% TEST (Futuro)**: Datos más nuevos para evaluación final

### 🕒 Metodología Temporal:

El split se basa en la variable **"DATA DE ABERTURA"** del dataset Sismetro:

1. **Ordenamiento temporal**: Todos los registros se ordenan por fecha de abertura
2. **División cronológica**: Split secuencial respetando el orden temporal
3. **Consistencia temporal**: Train < Val < Test en términos temporales

### 📁 Archivos Generados:

```
data/processed/
├── sismetro_inductive_train.pt    # Grafo de entrenamiento (pasado)
├── sismetro_inductive_val.pt      # Grafo de validación (presente) 
├── sismetro_inductive_test.pt     # Grafo de prueba (futuro)
└── sismetro_inductive_metadata.pt # Metadatos y períodos temporales
```

### 🎯 Características del Split Inductivo:

#### **Vocabularios Globales Consistentes**:
- Todos los splits usan los mismos mapeos de nodos
- Características de nodos basadas en información histórica acumulativa
- Total de nodos idéntico en todos los splits para compatibilidad

#### **Capacidad Inductiva**:
- **VAL**: Contiene nodos nuevos no vistos en TRAIN
- **TEST**: Contiene nodos nuevos no vistos en TRAIN ni VAL
- Simula escenarios reales donde aparecen nuevos patrimônios y localizações

#### **Preservación de Estructura**:
- ✅ Estructura bipartita mantenida en todos los splits
- ✅ Grafo no dirigido preservado
- ✅ Atributos de nodos consistentes

### 🚀 Comandos de Entrenamiento:

```bash
# Non-Contrastive Learning (recomendado para inductivo)
python src/train_nc.py --config_file src/config/inductive_sismetro.cfg

# GRACE
python src/train_grace.py --config_file src/config/inductive_sismetro.cfg

# Margin Loss  
python src/train_margin.py --config_file src/config/inductive_sismetro.cfg
```

### 📈 Estadísticas del Split:

**Distribución Temporal Real**: ~80/10/10
**Períodos**: Basados en fechas reales del dataset (2021-2024)
**Registros**: 16,929 registros totales divididos temporalmente
**Nodos**: 2,798 nodos totales (2,706 patrimônios + 92 localizações)

### 🔍 Capacidad Inductiva:

- **Validación**: Incluye nodos nuevos del "presente"
- **Test**: Incluye nodos nuevos del "futuro"
- **Escenario realista**: Simula predicción de enlaces para entidades nuevas

### 💡 Ventajas del Split Inductivo:

1. **Realismo temporal**: Respeta el orden cronológico real
2. **Evaluación robusta**: Testa capacidad de generalización a nodos nuevos
3. **Aplicabilidad práctica**: Simula uso en producción con datos futuros
4. **Consistencia metodológica**: Evita data leakage temporal

### 🛠️ Scripts de Verificación:

```bash
# Verificar integridad del split
python data_preparation/verify_sismetro_inductive.py

# Ejemplo de uso
python example_sismetro_inductive.py
```

### 📋 Metadatos Incluidos:

- **Períodos temporales**: Fechas exactas de cada split
- **Mapeos de nodos**: Vocabularios globales consistentes
- **Estadísticas**: Conteos y distribuciones por split
- **Información inductiva**: Análisis de nodos nuevos por período

¡El dataset Sismetro con split inductivo temporal está listo para experimentos avanzados de predicción de enlaces! 🎉
