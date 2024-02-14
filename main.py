import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time
import signal
from datetime import datetime

columns = ['Time', 'GPU Util (%)', 'Memory Util (%)', 'Used Memory (MiB)']
data = pd.DataFrame(columns=columns)
collecting = True

def signal_handler(sig, frame):
    global collecting
    print('Stopping data collection...')
    collecting = False

signal.signal(signal.SIGINT, signal_handler)

def collect_gpu_metrics():
    global data
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        current_time = pd.to_datetime('now')
        metrics = smi_output.strip().split(', ')
        metrics = [float(metric) for metric in metrics]
        return [current_time] + metrics
    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
        return []

print("Collecting GPU metrics. Press Ctrl+C to stop...")
while collecting:
    print(f"[{datetime.now().strftime('%H:%m.%f')}] Collecting ...")
    metrics = collect_gpu_metrics()
    if metrics:
        temp_df = pd.DataFrame([metrics], columns=columns)
        data = pd.concat([data, temp_df], ignore_index=True)
    time.sleep(1)

if not data.empty:
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(data['Time'], data['GPU Util (%)'], label='GPU Util (%)')
    plt.xlabel('Time')
    plt.ylabel('GPU Util (%)')
    plt.title('GPU Utilization Over Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(data['Time'], data['Memory Util (%)'], label='Memory Util (%)', color='orange')
    plt.plot(data['Time'], data['Used Memory (MiB)'], label='Used Memory (MiB)', color='green')
    plt.xlabel('Time')
    plt.ylabel('Memory Util (%) & Used Memory (MiB)')
    plt.title('Memory Utilization and Usage Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No data collected.")
