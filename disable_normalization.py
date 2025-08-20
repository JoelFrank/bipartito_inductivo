#!/usr/bin/env python3
"""
Script para desactivar toda la normalización L2 en eval.py
"""

import re

# Leer el archivo
with open('src/lib/eval.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Patrón para encontrar líneas de normalización
pattern = r'^(\s*)(.*torch\.nn\.functional\.normalize.*p=2.*dim=1.*)$'

# Reemplazar con versión comentada
def replace_normalize(match):
    indent = match.group(1)
    line = match.group(2)
    return f'{indent}# {line}  # DESACTIVADO: Normalization removed'

# Aplicar el reemplazo
new_content = re.sub(pattern, replace_normalize, content, flags=re.MULTILINE)

# También buscar el otro patrón (F.normalize)
pattern2 = r'^(\s*)(.*F\.normalize.*p=2.*dim=1.*)$'
new_content = re.sub(pattern2, replace_normalize, new_content, flags=re.MULTILINE)

# Escribir el archivo modificado
with open('src/lib/eval.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Normalización L2 desactivada en todas las líneas de eval.py")
print("📝 Las líneas ahora están comentadas con # DESACTIVADO")
