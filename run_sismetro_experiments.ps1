# Script para ejecutar todos los experimentos con Sismetro
# Comandos para Windows PowerShell

Write-Host "=== EXPERIMENTOS CON DATASET SISMETRO ===" -ForegroundColor Green

# Cambiar al directorio del proyecto
cd "C:\Users\joelf\Documents\non-contrastive-link-prediction-bipartite\src"

Write-Host "`n--- EXPERIMENTOS TRANSDUCTIVOS ---" -ForegroundColor Yellow

Write-Host "`n1. T-BGRL Original (Versão Bipartita) - Transductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_tbgrl_transductive.cfg

Write-Host "`n2. BGRL (Versão Bipartita) - Transductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_bgrl_transductive.cfg

Write-Host "`n3. GBT (Versão Bipartita) - Transductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_gbt_transductive.cfg

Write-Host "`n4. CCA-SSG (Versão Bipartita) - Transductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_cca_transductive.cfg

Write-Host "`n--- EXPERIMENTOS INDUCTIVOS ---" -ForegroundColor Yellow

Write-Host "`n5. T-BGRL Original (Versão Bipartita) - Inductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_tbgrl_inductive.cfg

Write-Host "`n6. BGRL (Versão Bipartita) - Inductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_bgrl_inductive.cfg

Write-Host "`n7. GBT (Versão Bipartita) - Inductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_gbt_inductive.cfg

Write-Host "`n8. CCA-SSG (Versão Bipartita) - Inductivo" -ForegroundColor Cyan
python train_nc.py --config_file config/sismetro_cca_inductive.cfg

Write-Host "`n=== TODOS LOS EXPERIMENTOS COMPLETADOS ===" -ForegroundColor Green
