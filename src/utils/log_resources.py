import os
import time
import psutil
import subprocess
import tensorflow as tf
from threading import Thread


def log_resources(log_dir: str, interval: int = 5, **kwargs) -> None:
    """
    Logs selected resources (CPU, RAM, GPU, CUDA, TensorFlow) at regular intervals.

    Args:
        log_dir (str): Directory to save log files.
        interval (int): Time interval between logs (seconds). Default is 5.
        kwargs: Resource options as boolean. Supported options: 
                "cpu", "ram", "gpu", "cuda", "tensorflow".
    """
    os.makedirs(log_dir, exist_ok=True)

    def log_cpu():
        log_path = os.path.join(log_dir, "cpu_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,CPU_Usage(%),Per-Core_Usage(%)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cpu_usage = psutil.cpu_percent()
                    per_core_usage = psutil.cpu_percent(percpu=True)
                    f.write(f"{timestamp},{cpu_usage},{','.join(map(str, per_core_usage))}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_ram():
        log_path = os.path.join(log_dir, "ram_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,Total(MB),Used(MB),Free(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    mem = psutil.virtual_memory()
                    total = mem.total / (1024 ** 2)
                    used = mem.used / (1024 ** 2)
                    free = mem.available / (1024 ** 2)
                    f.write(f"{timestamp},{total:.2f},{used:.2f},{free:.2f}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_gpu():
        log_path = os.path.join(log_dir, "gpu_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,GPU_ID,Memory_Used(MB),Memory_Total(MB),GPU_Utilization(%)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    gpu_stats = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    
                    for line in gpu_stats.split("\n"):
                        gpu_id, mem_used, mem_total, util = map(int, line.split(","))
                        f.write(f"{timestamp},{gpu_id},{mem_used},{mem_total},{util}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_cuda():
        log_path = os.path.join(log_dir, "cuda_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,Process_Memory_Used(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cuda_mem_stats = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    if cuda_mem_stats:
                        f.write(f"{timestamp},{cuda_mem_stats} MB\n")
                    else:
                        f.write(f"{timestamp},0 MB\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_tensorflow():
        log_path = os.path.join(log_dir, "tensorflow_usage_log.csv")
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("Timestamp,Device,Memory_Allocated(MB),Memory_Peak(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    gpus = tf.config.experimental.list_physical_devices("GPU")
                    for gpu in gpus:
                        device_name = gpu.name  # The correct device name
                        memory_info = tf.config.experimental.get_memory_info("GPU:0")
                        allocated_memory = memory_info["current"] / (1024 ** 2)  # Convert to MB
                        peak_memory = memory_info["peak"] / (1024 ** 2)  # Convert to MB
                        f.write(f"{timestamp},{device_name},{allocated_memory:.2f},{peak_memory:.2f}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    # Start logging threads based on user selection
    if kwargs.get("cpu", False):
        Thread(target=log_cpu, daemon=True).start()
    if kwargs.get("ram", False):
        Thread(target=log_ram, daemon=True).start()
    if kwargs.get("gpu", False):
        Thread(target=log_gpu, daemon=True).start()
    if kwargs.get("cuda", False):
        Thread(target=log_cuda, daemon=True).start()
    if kwargs.get("tensorflow", False):
        Thread(target=log_tensorflow, daemon=True).start()