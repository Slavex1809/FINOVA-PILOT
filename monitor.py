import psutil
import gc
import time
import numpy as np

class MemoryOptimizer:
    @staticmethod
    def get_memory_mb() -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024
    @staticmethod
    def clear_memory():
        gc.collect()

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {'inference_times': [], 'memory_usage': [], 'error_count': 0, 'success_count': 0}
    def time_execution(self, component_name: str):
        class Timer:
            def __init__(self, monitor, name):
                self.monitor, self.name, self.start_time = monitor, name, None
            def __enter__(self):
                self.start_time = time.time()
                return self
            def __exit__(self, *args):
                t = time.time() - self.start_time
                self.monitor._log_inference(self.name, t)
                self.monitor.metrics['success_count' if args[0] is None else 'error_count'] += 1
        return Timer(self, component_name)
    def _log_inference(self, c, t):
        self.metrics['inference_times'].append({'component': c, 'time': t})
    def log_memory_usage(self):
        self.metrics['memory_usage'].append({'usage_mb': MemoryOptimizer.get_memory_mb()})
    def get_health_report(self):
        it = [m['time'] for m in self.metrics['inference_times']]
        mu = [m['usage_mb'] for m in self.metrics['memory_usage']]
        return {
            'success_rate': self.metrics['success_count']/max(1, self.metrics['success_count']+self.metrics['error_count']),
            'avg_inference_time': np.mean(it) if it else 0.0,
            'avg_memory_usage': np.mean(mu) if mu else 0.0,
            'system_status': 'HEALTHY' if self.metrics['error_count']==0 else 'DEGRADED'
        }

# Глобальный экземпляр для импорта в других модулях
perf_monitor = PerformanceMonitor()
