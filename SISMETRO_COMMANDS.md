# Comandos individuales para ejecutar cada método con dataset Sismetro

## RESUMEN DE LAS LOCALIZAÇÕES:
# Es normal que no haya localizações nuevas en val/test porque:
# - El dataset tiene solo 92 localizações fijas (infraestructura de la empresa)
# - Estas localizações aparecen desde el inicio del período temporal
# - Los patrimônios (equipos) sí pueden aparecer nuevos con el tiempo
# - Esto refleja la realidad: las ubicaciones son fijas, los equipos se agregan

## EXPERIMENTOS TRANSDUCTIVOS (Grafo completo):

### 1. T-BGRL Original (Versão Bipartita) - Transductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_tbgrl_transductive.cfg
```

### 2. BGRL (Versão Bipartita) - Transductivo  
```bash
cd src
python train_nc.py --config_file config/sismetro_bgrl_transductive.cfg
```

### 3. GBT (Versão Bipartita) - Transductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_gbt_transductive.cfg
```

### 4. CCA-SSG (Versão Bipartita) - Transductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_cca_transductive.cfg
```

## EXPERIMENTOS INDUCTIVOS (Split temporal 80/10/10):

### 5. T-BGRL Original (Versão Bipartita) - Inductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_tbgrl_inductive.cfg
```

### 6. BGRL (Versão Bipartita) - Inductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_bgrl_inductive.cfg
```

### 7. GBT (Versão Bipartita) - Inductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_gbt_inductive.cfg
```

### 8. CCA-SSG (Versão Bipartita) - Inductivo
```bash
cd src
python train_nc.py --config_file config/sismetro_cca_inductive.cfg
```

## MÉTODOS ADICIONALES:

### 9. GRACE (baseline)
```bash
cd src
python train_grace.py --config_file config/sismetro_tbgrl_transductive.cfg
```

### 10. Margin Loss (baseline)
```bash
cd src
python train_margin.py --config_file config/sismetro_tbgrl_transductive.cfg
```

## EJECUTAR TODOS LOS EXPERIMENTOS:
```bash
# Windows PowerShell
.\run_sismetro_experiments.ps1

# Bash (Linux/Mac)
bash run_sismetro_experiments.sh
```

## RESULTADOS:
Los resultados se guardarán en:
- `runs/sismetro_[método]_[transductive|inductive]/`
- Logs, métricas y checkpoints del modelo

## ANÁLISIS:
Para analizar los resultados después:
```bash
python analyze_results.py --experiment_dir runs/
```
