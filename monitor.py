import time
import psutil
import pandas as pd
from contextlib import contextmanager


class OperatorMonitor:
    def __init__(self):
        self.data = []

    @contextmanager
    def monitor(self, model_name, operator_name, layer_number):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            yield  # 在这里执行算子代码
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            exec_time = end_time - start_time
            memory_used = end_memory - start_memory

            self.data.append({
                'model_name': model_name,
                'operator_name': operator_name,
                'memory_used': memory_used,
                'exec_time': exec_time,
                'layer_number': layer_number
            })

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)