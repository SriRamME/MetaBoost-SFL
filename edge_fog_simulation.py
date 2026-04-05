import time
import numpy as np

def simulate_edge_fog_inference(model, X_sample, n_samples=100):
    latencies = []
    for _ in range(n_samples):
        start = time.perf_counter()
        _ = model.predict(X_sample)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    mean_latency = np.mean(latencies)
    print(f"Simulated Edge-Fog Inference Latency: {mean_latency:.2f} ms")
    print(f"92% predictions under 100ms: {np.percentile(latencies, 92):.2f} ms")
    return mean_latency
