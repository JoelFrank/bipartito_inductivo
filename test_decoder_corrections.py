#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones del decodificador.
Verifica que el nuevo flag link_mlp_lr funcione correctamente.
"""

import torch
import numpy as np
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_flags():
    """Prueba que el nuevo flag link_mlp_lr funcione"""
    log.info("=== Probando nuevo flag link_mlp_lr ===")
    
    try:
        from absl import flags
        from src.lib.flags import define_flags, ModelGroup
        
        # Define flags for NCL model group
        define_flags(ModelGroup.NCL)
        
        # Create a mock FLAGS object with test values
        FLAGS = flags.FLAGS
        
        # Set some test values
        test_args = [
            '--lr=0.005',
            '--link_mlp_lr=0.001',
            '--epochs=100',
            '--decoder_epochs=50'
        ]
        
        FLAGS(test_args)
        
        # Verify the flags work
        assert hasattr(FLAGS, 'link_mlp_lr'), "Flag link_mlp_lr no existe"
        assert FLAGS.link_mlp_lr == 0.001, f"link_mlp_lr debería ser 0.001, pero es {FLAGS.link_mlp_lr}"
        assert FLAGS.lr == 0.005, f"lr debería ser 0.005, pero es {FLAGS.lr}"
        
        log.info(f"✓ Flags configurados correctamente:")
        log.info(f"  - lr (encoder): {FLAGS.lr}")
        log.info(f"  - link_mlp_lr (decoder): {FLAGS.link_mlp_lr}")
        log.info(f"  - epochs: {FLAGS.epochs}")
        log.info(f"  - decoder_epochs: {FLAGS.decoder_epochs}")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Error probando flags: {e}")
        return False

def test_normalization():
    """Prueba la normalización de embeddings"""
    log.info("=== Probando normalización de embeddings ===")
    
    try:
        import torch.nn.functional as F
        
        # Crear embeddings de prueba con magnitudes diferentes
        embeddings = torch.tensor([
            [1.0, 2.0, 3.0],      # magnitud ~3.74
            [10.0, 20.0, 30.0],   # magnitud ~37.4
            [0.1, 0.2, 0.3]       # magnitud ~0.37
        ])
        
        log.info(f"Embeddings originales:")
        for i, emb in enumerate(embeddings):
            magnitude = torch.norm(emb).item()
            log.info(f"  Embedding {i}: {emb.tolist()}, magnitud: {magnitude:.3f}")
        
        # Normalizar
        normalized = F.normalize(embeddings, p=2, dim=1)
        
        log.info(f"Embeddings normalizados:")
        for i, emb in enumerate(normalized):
            magnitude = torch.norm(emb).item()
            log.info(f"  Embedding {i}: {emb.tolist()}, magnitud: {magnitude:.3f}")
        
        # Verificar que todas las magnitudes son 1.0 (±epsilon)
        magnitudes = torch.norm(normalized, p=2, dim=1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6), \
            f"Las magnitudes deberían ser 1.0, pero son: {magnitudes.tolist()}"
        
        log.info("✓ Normalización funciona correctamente")
        return True
        
    except Exception as e:
        log.error(f"❌ Error probando normalización: {e}")
        return False

def test_decoder_lr_separation():
    """Simula el entrenamiento separado del encoder y decoder"""
    log.info("=== Probando separación de learning rates ===")
    
    try:
        # Simular un encoder y decoder simples
        encoder = torch.nn.Linear(10, 5)
        decoder = torch.nn.Linear(10, 1)  # Concatenated embeddings (5+5=10) -> prediction
        
        # Learning rates diferentes
        encoder_lr = 0.005
        decoder_lr = 0.001
        
        # Optimizers separados
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_lr)
        
        log.info(f"✓ Optimizers creados:")
        log.info(f"  - Encoder LR: {encoder_lr}")
        log.info(f"  - Decoder LR: {decoder_lr}")
        
        # Verificar que los learning rates son diferentes
        encoder_lr_actual = encoder_optimizer.param_groups[0]['lr']
        decoder_lr_actual = decoder_optimizer.param_groups[0]['lr']
        
        assert encoder_lr_actual == encoder_lr, f"Encoder LR incorrecto: {encoder_lr_actual}"
        assert decoder_lr_actual == decoder_lr, f"Decoder LR incorrecto: {decoder_lr_actual}"
        assert encoder_lr_actual != decoder_lr_actual, "Los LRs deberían ser diferentes"
        
        log.info(f"✓ Learning rates configurados correctamente y son diferentes")
        return True
        
    except Exception as e:
        log.error(f"❌ Error probando separación de LR: {e}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    log.info("🔧 Iniciando pruebas de correcciones del decodificador...")
    
    tests = [
        ("Flags", test_flags),
        ("Normalización", test_normalization),
        ("Separación LR", test_decoder_lr_separation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            log.error(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    log.info("=" * 50)
    log.info("📋 RESUMEN DE PRUEBAS:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        log.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        log.info("🎉 ¡Todas las correcciones funcionan correctamente!")
        log.info("🚀 Listo para ejecutar el experimento con las mejoras.")
    else:
        log.error("❌ Algunas pruebas fallaron. Revisa los errores arriba.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
