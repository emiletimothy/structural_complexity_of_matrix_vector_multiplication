# Structural Complexity of Matrix-Vector Multiplication

A research implementation exploring **MST-based compression** for efficient matrix-vector multiplication, with a focus on the relationship between **VC dimension** and algorithmic performance.

## Overview

This project implements novel algorithms for computing matrix-vector products `Mv` using minimum spanning tree (MST) compression. Rather than storing the full matrix `M`, we construct a Δ-labeled spanning tree that exploits structural similarity between matrix rows/columns to achieve faster query times.

## Algorithms

### 1. MST(M^T) - Row-wise MST
- Builds MST over matrix rows using Hamming distance
- Computes `Mv` via top-down tree traversal
- **Query time**: O(weight(T) + n + m)
- **Best for**: Matrices with similar rows

### 2. MST(M) - Column-wise MST  
- Builds MST over matrix columns using Hamming distance
- Uses path-Δ matrix representation (Lemma 1)
- Computes via bottom-up subtree sum aggregation
- **Query time**: O(weight(T) + n)
- **Best for**: Matrices with similar columns

The `MSTCompressor` automatically selects the optimal algorithm by computing both MSTs during preprocessing and choosing the one with lower weight.

## Installation

```bash
# Clone the repository
git clone https://github.com/emiletimothy/structural_complexity_of_matrix_vector_multiplication.git
cd structural_complexity_of_matrix_vector_multiplication

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Core numerical operations
- `numba` - JIT compilation for performance-critical paths
- `matplotlib` - Visualization for experiments
- `scipy`, `pandas`, `seaborn` - Data analysis
- Additional dependencies for extended experiments (PyTorch, Transformers)

## Usage

### Basic Usage

```python
import numpy as np
from fast_mat_vec_multiplication import MSTCompressor

# Create a binary matrix
M = np.random.randint(0, 2, size=(1000, 500))

# Preprocess: Build MST
compressor = MSTCompressor(M)

# Query: Compute Mv efficiently
v = np.random.randn(500)
result = compressor.matvec(v)

# Get compression statistics
stats = compressor.get_compression_stats()
print(f"Algorithm: {stats['algorithm']}")
print(f"Compression ratio: {stats['compression_ratio']:.2%}")
print(f"Tree weight: {stats['tree_weight']}")
```

### Force Specific Algorithm

```python
# Force row-wise MST
compressor_row = MSTCompressor(M, use_column_mst=False)

# Force column-wise MST  
compressor_col = MSTCompressor(M, use_column_mst=True)

# Auto-select (default)
compressor_auto = MSTCompressor(M, use_column_mst=None)
```

## VC Dimension Experiments

The `vc_dim_experiments.py` script explores the relationship between **Vapnik-Chervonenkis (VC) dimension** and MST-based algorithm performance.

### Running Experiments

```python
python vc_dim_experiments.py
```

This will:
1. Generate matrices with controlled VC dimensions (2 to 12)
2. Build both row and column MSTs
3. Benchmark query times vs NumPy baseline
4. Generate plots showing:
   - Tree weight vs VC dimension (row MST)
   - Tree weight vs VC dimension (column MST)
   - Runtime comparison vs VC dimension

### Key Functions

- `generate_matrix_of_vc_dimension_d(n, d)` - Generates n×n binary matrix with VC dimension ≈ d
- `compute_vc_dimension(M)` - Computes VC dimension of matrix M
- `time_mat_vec_processes(M)` - Benchmarks MST vs NumPy for matrix M

## Performance Characteristics

### Preprocessing
- **Time**: O(m²n) for row MST or O(n²m) for column MST
- **Space**: O(weight(T)) - stores only MST edges with sparse deltas
- **One-time cost**: Amortized over multiple queries

### Query Time
- **MST(M^T)**: O(weight(T) + n + m)
- **MST(M)**: O(weight(T) + n)
- **NumPy baseline**: O(mn)

### When MST Wins
MST-based multiplication is faster when:
- `weight(T) << mn` (low structural complexity)
- Multiple queries on same matrix (amortize preprocessing)
- Matrix exhibits row/column similarity

### Optimizations
- **Numba JIT compilation** for hot paths
- **Sparse delta storage** (only non-zero entries)
- **Iterative DFS** (avoids recursion overhead)
- **Contiguous memory layout** for cache efficiency

## File Structure

```
.
├── fast_mat_vec_multiplication.py  # Core MST compression algorithms
│   ├── SparseDelta                 # Sparse vector representation
│   ├── DeltaLabeledEdge           # MST edge with delta label
│   ├── DeltaLabeledSpanningTree   # Tree structure
│   └── MSTCompressor              # Main API
│
├── vc_dim_experiments.py          # VC dimension experiments
│   ├── generate_matrix_of_vc_dimension_d()
│   ├── compute_vc_dimension()
│   └── time_mat_vec_processes()
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Theoretical Background

### MST Compression
The algorithms exploit the observation that structurally simple matrices (rows/columns with high similarity) yield MSTs with low total weight. By storing only the **deltas** (differences) along MST edges, we compress the matrix representation.

### Path-Δ Matrix (Lemma 1)
For column-wise MST, the key insight is that the original matrix M can be reconstructed as:
```
M = N^(T) where N^(T) is the path-Δ matrix
```
This allows computing `Mv = N^(T)v` via:
1. Bottom-up aggregation: σ = subtree sums of v
2. Delta multiplication: result = Δσ

### VC Dimension
The **Vapnik-Chervonenkis dimension** measures the complexity of a set system (or matrix). Our experiments investigate how VC dimension correlates with MST weight, providing insight into which matrix structures benefit most from MST compression.

## Experimental Results

Run experiments to observe:
- **Low VC dimension** → Lower MST weights → Faster queries
- **High VC dimension** → Higher MST weights → May not beat NumPy
- **Algorithm selection** matters: row vs column MST depends on matrix structure

## Future Work

- Approximate MST for faster preprocessing
- Extension to non-binary matrices
- Online/streaming MST updates

## Citation

If you use this code in your research, please cite:
```
@article{anand2025structuralcomplexitymatrixvectormultiplication,
      title={The Structural Complexity of Matrix-Vector Multiplication}, 
      author={Emile Anand and Jan van den Brand and Rose McCarty},
      year={2025},
      eprint={2502.21240},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2502.21240}, 
}```
