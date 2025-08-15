import json
import os
import time
from datetime import datetime
import logging

class LocalLogger:
    """
    Reemplazo simple para wandb que guarda métricas localmente
    """
    def __init__(self, project=None, config=None, name=None):
        self.project = project
        self.config = config or {}
        self.name = name
        self.metrics = []
        self.start_time = time.time()
        
        # Generar un ID único simple
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.id = self.run_id  # Alias para compatibilidad con wandb
        
        # Crear directorio para este run
        self.log_dir = os.path.join("logs", self.project or "default", self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Guardar configuración inicial
        if self.config:
            config_path = os.path.join(self.log_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        
        self.logger = logging.getLogger(f"{self.project}_{self.run_id}")
        self.logger.info(f"Iniciando run: {self.run_id}")
        
    def log(self, metrics_dict):
        """Log métricas con timestamp"""
        timestamp = time.time() - self.start_time
        entry = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            **metrics_dict
        }
        self.metrics.append(entry)
        
        # Log a archivo inmediatamente
        metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Log a consola
        metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics_dict.items()])
        self.logger.info(f"Step {len(self.metrics)}: {metric_str}")
    
    def finish(self):
        """Finalizar el run y guardar resumen"""
        summary_path = os.path.join(self.log_dir, "summary.json")
        summary = {
            'run_id': self.run_id,
            'project': self.project,
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_steps': len(self.metrics),
            'config': self.config
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Run finalizado: {self.run_id}")

class MockWandb:
    """
    Mock object que reemplaza wandb sin cambiar la API
    """
    run = None
    
    @staticmethod
    def init(project=None, config=None, name=None):
        MockWandb.run = LocalLogger(project=project, config=config, name=name)
        return MockWandb.run
    
    @staticmethod
    def log(metrics_dict):
        if MockWandb.run:
            MockWandb.run.log(metrics_dict)
    
    @staticmethod
    def finish():
        if MockWandb.run:
            MockWandb.run.finish()
            MockWandb.run = None

# Crear alias para usar como wandb
wandb = MockWandb()
