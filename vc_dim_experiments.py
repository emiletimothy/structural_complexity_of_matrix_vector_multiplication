import numpy as np
import random
from itertools import combinations
import time
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from fast_mat_vec_multiplication import MSTCompressor

def generate_matrix_of_vc_dimension_d(n, d):
    planes = []
    points = []
    for i in range(n):
        weights = [np.random.rand() for _ in range(d)]
        bias = np.random.rand()
        planes.append([weights,bias])
        point = [np.random.rand() for _ in range(d)]
        points.append(point)
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i][j] = 1*(np.dot(planes[i][0], points[j]) > d * 0.5 * planes[i][1])
    return M

def compute_vc_dimension(M):    
    n = M.shape[0]
    
    def is_shattered_fast(col_subset):
        k = len(col_subset)
        if k > 20:
        
            return False
        submatrix = M[:, col_subset]
        seen = set()
        for i in range(len(submatrix)):
            row_hash = hash(submatrix[i].tobytes())
            if row_hash in seen:
                continue
            seen.add(row_hash)
            if len(seen) > 2**k:
                return False
        
        return len(seen) == 2**k
    
    for k in range(min(n, 20), 0, -1):
        for col_subset in combinations(range(n), k):
            if is_shattered_fast(col_subset):
                return k
    return 0
    
def time_mat_vec_processes(M):
    t0 = time.perf_counter()
    compressor = MSTCompressor(M)
    prep_time_single = time.perf_counter() - t0
    print(f"Preprocessing time: {prep_time_single:.4f}s")

    num_queries = 1000
    num_warmup = 50
    query_vectors = [np.random.randn(n) for _ in range(num_queries + num_warmup)]
    
    # Warmup phase for MST (untimed)
    for i in range(num_warmup):
        _ = compressor.matvec(query_vectors[i])
    
    # Collect individual MST query times
    mst_query_times = [None] * num_queries
    for i in range(num_queries):
        t0 = time.perf_counter()
        _ = compressor.matvec(query_vectors[num_warmup + i])
        mst_query_times[i] = (time.perf_counter() - t0) * 1000  # in ms

    # Warmup phase for NumPy (untimed)
    for i in range(num_warmup):
        _ = M @ query_vectors[i]
    
    # Collect individual numpy query times
    numpy_query_times = [None] * num_queries
    for i in range(num_queries):
        t0 = time.perf_counter()
        _ = M @ query_vectors[num_warmup + i]
        numpy_query_times[i] = (time.perf_counter() - t0) * 1000  # in ms

    avg_mst_time = np.mean(mst_query_times)
    avg_numpy_time = np.mean(numpy_query_times)
    
    print(f"Query time single-level MST: {avg_mst_time:.4f}ms (avg over {num_queries} queries)")
    print(f"Query time numpy: {avg_numpy_time:.4f}ms (avg over {num_queries} queries)")

    return compressor.tree_column.weight, avg_mst_time, avg_numpy_time, mst_query_times, numpy_query_times

n=4096
mst_column_weights = []
mst_times = []
numpy_times = []

max_d = 12
all_mst_runtimes = {}
all_numpy_runtimes = {}
all_mst_column_weights = {}
 
for d in range(2, max_d+1):
    print("VC dimension = ", d)
    M = generate_matrix_of_vc_dimension_d(n,d-1)
    mst_column_weight, mst_time, numpy_time, mst_query_times, numpy_query_times = time_mat_vec_processes(M)
    mst_column_weights.append(mst_column_weight)
    print("MST column weight = ", mst_column_weight)
    mst_times.append(mst_time)
    numpy_times.append(numpy_time)
    all_mst_runtimes[d] = mst_query_times
    all_numpy_runtimes[d] = numpy_query_times
    all_mst_column_weights[d] = mst_column_weight

with open('query_runtimes.txt', 'w') as f:
    f.write("Individual Query Runtimes (in milliseconds)\n")
    f.write("="*60 + "\n\n")
    
    for d, mst_times_for_d in all_mst_runtimes.items():
        sorted_times = sorted(mst_times_for_d)
        f.write(f"VC Dimension {d} - MST Query Times:\n")
        f.write(f"  MST Column Weight: {all_mst_column_weights[d]}\n")
        f.write(f"  Mean: {np.mean(mst_times_for_d):.4f}ms\n")
        f.write(f"  Std: {np.std(mst_times_for_d):.4f}ms\n")
        f.write(f"  Min: {np.min(mst_times_for_d):.4f}ms\n")
        f.write(f"  Max: {np.max(mst_times_for_d):.4f}ms\n")
        f.write(f"  All times (sorted): {sorted_times}\n\n")
    
    f.write("\n" + "="*60 + "\n\n")
    
    for d, numpy_times_for_d in all_numpy_runtimes.items():
        sorted_times = sorted(numpy_times_for_d)
        f.write(f"VC Dimension {d} - NumPy Query Times:\n")
        f.write(f"  Mean: {np.mean(numpy_times_for_d):.4f}ms\n")
        f.write(f"  Std: {np.std(numpy_times_for_d):.4f}ms\n")
        f.write(f"  Min: {np.min(numpy_times_for_d):.4f}ms\n")
        f.write(f"  Max: {np.max(numpy_times_for_d):.4f}ms\n")
        f.write(f"  All times (sorted): {sorted_times}\n\n")

print("\nAll individual query runtimes saved to 'query_runtimes.txt'")
 

plt.figure()
plt.title("Runtimes vs VC-dimension")
plt.plot([i for i in range(2,max_d+1)], numpy_times, color="black", label="NumPy")
plt.plot([i for i in range(2,max_d+1)], mst_times, color="blue", label="Our MST-based algorithm")
plt.xlabel("VC dimension")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig("Runtimes_vc_dim_parameterization.png")
plt.show()


with open("query_runtimes.txt", "r") as f:
    lines = f.readlines()

mst_data = {}  # {vc_dim: {'mean': x, 'min': y, 'max': z}}
numpy_data = {}

i = 0
while i < len(lines):
    line = lines[i]
    
    if "VC Dimension" in line and "MST Query Times" in line:
        vc_dim = int(line.split()[2])
        # Skip the MST Column Weight line (i+1), then read Mean, Std, Min, Max
        mean = float(lines[i+2].split()[1].replace('ms', ''))
        std = float(lines[i+3].split()[1].replace('ms', ''))
        min_val = float(lines[i+4].split()[1].replace('ms', ''))
        max_val = float(lines[i+5].split()[1].replace('ms', ''))
        mst_data[vc_dim] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
        i += 6
    elif "VC Dimension" in line and "NumPy Query Times" in line:
        vc_dim = int(line.split()[2])
        # Read next 4 lines for Mean, Std, Min, Max
        mean = float(lines[i+1].split()[1].replace('ms', ''))
        std = float(lines[i+2].split()[1].replace('ms', ''))
        min_val = float(lines[i+3].split()[1].replace('ms', ''))
        max_val = float(lines[i+4].split()[1].replace('ms', ''))
        numpy_data[vc_dim] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
        i += 5
    else:
        i += 1

# Extract data for plotting
vc_dims = sorted(mst_data.keys())
mst_means = [mst_data[d]['mean'] for d in vc_dims]
mst_stds = [mst_data[d]['std'] for d in vc_dims]
mst_mins = [mst_data[d]['min'] for d in vc_dims]
mst_maxs = [mst_data[d]['max'] for d in vc_dims]

numpy_means = [numpy_data[d]['mean'] for d in vc_dims]
numpy_stds = [numpy_data[d]['std'] for d in vc_dims]
numpy_mins = [numpy_data[d]['min'] for d in vc_dims]
numpy_maxs = [numpy_data[d]['max'] for d in vc_dims]

# Calculate error bars (distance from mean to min/max)
mst_err_lower = [mean - min_val for mean, min_val in zip(mst_means, mst_mins)]
mst_err_upper = [max_val - mean for mean, max_val in zip(mst_means, mst_maxs)]
numpy_err_lower = [mean - min_val for mean, min_val in zip(numpy_means, numpy_mins)]
numpy_err_upper = [max_val - mean for mean, max_val in zip(numpy_means, numpy_maxs)]


# Plotting
plt.figure(figsize=(10, 6))
plt.title("Mean Runtime vs VC-dimension (with Min/Max error bars)")
plt.errorbar(vc_dims, mst_means, yerr=[mst_err_lower, mst_err_upper], 
             fmt='o', color="blue", label="Our MST-based algorithm", capsize=5)
plt.errorbar(vc_dims, numpy_means, yerr=[numpy_err_lower, numpy_err_upper], 
             fmt='o', color="black", label="NumPy", capsize=5)
plt.xlabel("VC dimension")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("MinMaxRuntime_vc_dim_parameterization.png", dpi=300, bbox_inches='tight')
plt.show()

# Second plot with ±1 std error bars
plt.figure(figsize=(10, 6))
plt.title("Mean Runtime vs VC-dimension (with ±1 std error bars)")
plt.errorbar(vc_dims, mst_means, yerr=mst_stds, 
             fmt='o', color="blue", label="Our MST-based algorithm", capsize=5)
plt.errorbar(vc_dims, numpy_means, yerr=numpy_stds, 
             fmt='o', color="black", label="NumPy", capsize=5)
plt.xlabel("VC dimension")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("StdRuntime_vc_dim_parameterization.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot MST column weights vs VC dimension
plt.figure(figsize=(10, 6))
plt.title("MST Column Weight vs VC-dimension")
mst_column_weight_values = [all_mst_column_weights[d] for d in vc_dims]
plt.plot(vc_dims, mst_column_weight_values, 'o-', color="green", linewidth=2, markersize=8)
plt.xlabel("VC dimension")
plt.ylabel("MST Column Weight")
plt.grid(True, alpha=0.3)
plt.savefig("MST_Column_Weight_vc_dim.png", dpi=300, bbox_inches='tight')
plt.show()

