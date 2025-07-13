import contextvars
import time
import logging
from functools import wraps

# 共通の data_id 用 ContextVar
data_id_var = contextvars.ContextVar("data_id", default="unknown")

# ログ設定（必要に応じてファイル名などを調整）
logging.basicConfig(
    filename="execution_time.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        data_id = data_id_var.get()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start

        module_name = func.__module__
        function_name = func.__qualname__  # クラスメソッドなども対応
        logging.info(f"[ID={data_id}] {module_name}.{function_name} executed in {duration:.4f}s")

        return result
    return wrapper