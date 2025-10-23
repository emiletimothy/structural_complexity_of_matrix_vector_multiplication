from typing import Dict, List, Tuple
from collections import defaultdict, deque
import numpy as np
from numba import njit

@njit
def _sparse_dot_jit(indices: np.ndarray, values: np.ndarray, v: np.ndarray) -> float:
    """JIT-compiled sparse dot product: Δ · v."""
    result = 0.0
    for i in range(len(indices)):
        result += values[i] * v[indices[i]]
    return result

@njit
def _column_mst_sigma_jit(v_extended: np.ndarray, children_list_data: np.ndarray,
                           children_list_ptr: np.ndarray, postorder: np.ndarray,
                           sigma_values: np.ndarray, subtree_sum: np.ndarray) -> None:
    """JIT-compiled bottom-up sigma computation for column MST."""
    for idx in range(len(postorder) - 1, -1, -1):
        node = postorder[idx]
        total = subtree_sum[node]
        start = children_list_ptr[node]
        end = children_list_ptr[node + 1]
        for i in range(start, end):
            child = children_list_data[i, 0]
            edge_idx = children_list_data[i, 1]
            child_sum = subtree_sum[child]
            sigma_values[edge_idx] = child_sum
            total += child_sum
        subtree_sum[node] = total


@njit
def _column_mst_result_jit(delta_indices: np.ndarray, delta_values: np.ndarray,
                            sigma_repeated: np.ndarray, result: np.ndarray) -> None:
    """JIT-compiled result accumulation for column MST."""
    for i in range(len(delta_indices)):
        result[delta_indices[i]] += delta_values[i] * sigma_repeated[i]


class SparseDelta:
    """Sparse representation of delta vector Δ = M_child - M_parent."""
    
    def __init__(self, delta: np.ndarray):
        """Store only non-zero entries of delta vector."""
        self.nonzero_indices = np.nonzero(delta)[0].astype(np.int64)
        self.nonzero_values = delta[self.nonzero_indices].astype(np.float64)
        self.nnz = len(self.nonzero_indices)  # Number of non-zeros (Hamming distance)
    
    def dot(self, v: np.ndarray) -> float:
        """Compute Δ · v efficiently using only non-zero entries."""
        if self.nnz == 0:
            return 0.0
        # Use JIT-compiled version for better performance
        return _sparse_dot_jit(self.nonzero_indices, self.nonzero_values, v)


class DeltaLabeledEdge:
    """Directed edge (parent → child) with delta label Δ = M_child - M_parent."""
    
    def __init__(self, parent: int, child: int, delta: np.ndarray):
        self.parent = parent
        self.child = child
        self.sparse_delta = SparseDelta(delta)
        self.weight = self.sparse_delta.nnz  # nnz(Δ_e)
    
    def __repr__(self):
        return f"Edge({self.parent}→{self.child}, nnz={self.weight})"

class DeltaLabeledSpanningTree:
    """Δ-labeled spanning tree for M^T (operates on rows of M)."""
    
    def __init__(self, root: int, edges: List[DeltaLabeledEdge], num_vertices: int):
        self.root = root
        self.edges = edges
        self.num_vertices = num_vertices
        self.weight = sum(edge.weight for edge in edges)  # Total weight = Σ nnz(Δ_e)
        
        # Build adjacency list: parent -> [(child, sparse_delta), ...]
        self.children_map = defaultdict(list)
        for edge in edges:
            self.children_map[edge.parent].append((edge.child, edge.sparse_delta))
    
    def __repr__(self):
        return f"DeltaLabeledSpanningTree(root={self.root}, vertices={self.num_vertices}, weight={self.weight})"


class MSTCompressor:
    """MST-based compression for fast matrix-vector multiplication.
    
    Supports two algorithms:
    1. MST(M^T): Row-wise MST for computing Mv
    2. MST(M): Column-wise MST for computing Mv (path-Δ approach)
    """
    
    def __init__(self, matrix: np.ndarray, use_column_mst: bool = None):
        """
        Preprocess matrix M by building MST.
        
        Args:
            matrix: Binary matrix M of shape (m, n)
            use_column_mst: If True, force column MST. If False, force row MST.
                          If None (default), compute both and choose the one with lower weight.
        """
        self.original_matrix = matrix.copy()
        self.m, self.n = matrix.shape
        self.matrix = matrix.copy()
        self.tree_row = None
        self.tree_column = None
        self.use_column_mst = None  # Will be set after comparison
        self.row_root_sparse = None
        self.column_children_list = None
        self.column_postorder = None
        self.column_delta_indices = None
        self.column_delta_values = None
        self.column_edge_ptr = None
        self.column_edge_counts = None
        self.row_mst_weight = None
        self.column_mst_weight = None

        self._build_column_mst_with_zero_root()
        self.use_column_mst = True
    
    def _build_column_mst_with_zero_root(self):
        """Build MST(M) with zero root column prepended."""
        zero_col = np.zeros((self.m, 1), dtype=self.matrix.dtype)
        matrix_extended = np.hstack([zero_col, self.matrix])
        n_extended = self.n + 1
        
        # Build complete graph with Hamming distances between columns
        edges = []
        for i in range(n_extended):
            for j in range(i + 1, n_extended):
                weight = self._compute_hamming_distance(matrix_extended[:, i], matrix_extended[:, j])
                edges.append((weight, i, j))
        
        # Find MST
        edges_sorted = sorted(edges)
        parent = list(range(n_extended))
        rank = [0] * n_extended
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        mst_edges = []
        for weight, u, v in edges_sorted:
            if union(u, v):
                mst_edges.append((weight, u, v))
                if len(mst_edges) == n_extended - 1:
                    break
        
        # Root at column 0 (zero column)
        root = 0
        adj = defaultdict(list)
        for weight, u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS to build parent map
        parent_map = {root: root}
        queue = deque([root])
        visited = {root}
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    parent_map[neighbor] = current
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Build delta edges
        delta_edges = []
        for child, parent_col in parent_map.items():
            if child != parent_col:
                delta = matrix_extended[:, child] - matrix_extended[:, parent_col]
                edge = DeltaLabeledEdge(parent_col, child, delta)
                delta_edges.append(edge)
        
        self.tree_column = DeltaLabeledSpanningTree(root, delta_edges, n_extended)
        self.matrix_extended = matrix_extended
        self.n_extended = n_extended

        num_edges = len(delta_edges)
        self.column_children_list = [[] for _ in range(n_extended)]
        for idx, edge in enumerate(delta_edges):
            self.column_children_list[edge.parent].append((edge.child, idx))

        postorder = []
        stack = [self.tree_column.root]
        while stack:
            node = stack.pop()
            postorder.append(node)
            if node < len(self.column_children_list):
                for child, _ in self.column_children_list[node]:
                    stack.append(child)
        self.column_postorder = np.array(postorder, dtype=np.int64)

        # Prepare JIT-friendly data structures for column MST
        # Convert children_list to contiguous arrays
        total_children = sum(len(children) for children in self.column_children_list)
        self.column_children_data = np.empty((total_children, 2), dtype=np.int64)
        self.column_children_ptr = np.empty(n_extended + 1, dtype=np.int64)
        pos = 0
        self.column_children_ptr[0] = 0
        for node_idx in range(n_extended):
            for child, edge_idx in self.column_children_list[node_idx]:
                self.column_children_data[pos, 0] = child
                self.column_children_data[pos, 1] = edge_idx
                pos += 1
            self.column_children_ptr[node_idx + 1] = pos

        total_nonzeros = sum(edge.sparse_delta.nnz for edge in delta_edges)
        self.column_delta_indices = np.empty(total_nonzeros, dtype=np.int64)
        self.column_delta_values = np.empty(total_nonzeros, dtype=np.float64)
        self.column_edge_ptr = np.empty(num_edges + 1, dtype=np.int64)
        pos = 0
        self.column_edge_ptr[0] = 0
        for idx, edge in enumerate(delta_edges):
            nnz = edge.sparse_delta.nnz
            if nnz > 0:
                self.column_delta_indices[pos:pos + nnz] = edge.sparse_delta.nonzero_indices
                self.column_delta_values[pos:pos + nnz] = edge.sparse_delta.nonzero_values
                pos += nnz
            self.column_edge_ptr[idx + 1] = pos
        self.column_edge_counts = np.diff(self.column_edge_ptr)
    
    def _compute_hamming_distance(self, row1: np.ndarray, row2: np.ndarray) -> int:
        """Compute Hamming distance between two rows."""
        return np.count_nonzero(row1 != row2)
    
    def _build_distance_graph(self) -> List[Tuple[int, int, int]]:
        """Build complete graph with Hamming distances as edge weights."""
        edges = []
        for i in range(self.m):
            for j in range(i + 1, self.m):
                weight = self._compute_hamming_distance(self.matrix[i, :], self.matrix[j, :])
                edges.append((weight, i, j))
        return edges
    
    def _find_mst_kruskal(self, edges: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Find MST using Kruskal's algorithm with Union-Find."""
        edges_sorted = sorted(edges)  # Sort by weight
        parent = list(range(self.m))
        rank = [0] * self.m
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        mst_edges = []
        for weight, u, v in edges_sorted:
            if union(u, v):
                mst_edges.append((weight, u, v))
                if len(mst_edges) == self.m - 1:
                    break
        return mst_edges
    
    def _root_mst(self, mst_edges: List[Tuple[int, int, int]], root: int) -> Dict[int, int]:
        """Root the MST at specified vertex using BFS. Returns parent mapping."""
        # Build adjacency list (undirected)
        adj = defaultdict(list)
        for weight, u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS to assign parents
        parent = {root: root}  # Root is its own parent
        queue = deque([root])
        visited = {root}
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    parent[neighbor] = current
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return parent
    
    def _build_delta_labeled_edges(self, parent_map: Dict[int, int]) -> List[DeltaLabeledEdge]:
        """Build Δ-labeled edges from parent mapping."""
        edges = []
        for child, parent in parent_map.items():
            if child != parent:  # Skip root
                # Δ_{parent,child} = M_child - M_parent (difference of rows)
                delta = self.matrix[child, :] - self.matrix[parent, :]
                edge = DeltaLabeledEdge(parent, child, delta)
                edges.append(edge)
        return edges
    
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Compute Mv efficiently using Δ-labeled spanning tree.
        
        Args:
            v: Input vector of length n
            
        Returns:
            Result vector Mv of length m
        """
        if len(v) != self.n:
            raise ValueError(f"Vector length {len(v)} must match matrix columns {self.n}")
        
        return self._matvec_column_mst(v)
    
    def _matvec_column_mst(self, v: np.ndarray) -> np.ndarray:
        """
        MST(M) algorithm: Path-Δ matrix approach via Lemma 1.
        
        Key insight: Mv = N^{(T)} v where N^{(T)} is path-Δ matrix.
        By Lemma 1: N^{(T)} v = Δσ where:
          - σ_{(parent,child)} = sum of v_z for all z in subtree rooted at child
          - Δ is matrix with delta labels as columns
        
        Algorithm:
        1. Extend v with zero for root column
        2. Compute σ ∈ R^|E| via bottom-up DP: O(n) time
        3. Compute Mv = Δσ: O(weight(T)) time
        
        Time: O(weight(T) + n)
        """
        # Extend vector with zero for root column
        v_extended = np.zeros(self.n_extended)
        v_extended[1:] = v  # Root gets 0, original v at positions 1..n
        
        # Step 1: Compute σ via bottom-up DP on tree (JIT-compiled)
        # For each edge (parent→child): σ_{edge} = sum_{z in subtree(child)} v_z
        
        sigma_values = np.zeros(len(self.tree_column.edges), dtype=np.float64)
        subtree_sum = v_extended.astype(np.float64).copy()

        if hasattr(self, 'column_children_data') and self.column_children_data is not None:
            # Use JIT-compiled optimized path
            _column_mst_sigma_jit(v_extended.astype(np.float64), 
                                   self.column_children_data,
                                   self.column_children_ptr, 
                                   self.column_postorder,
                                   sigma_values, 
                                   subtree_sum)
        else:
            # Fallback to Python loop
            for node in reversed(self.column_postorder):
                total = subtree_sum[node]
                if node < len(self.column_children_list):
                    for child, edge_idx in self.column_children_list[node]:
                        child_sum = subtree_sum[child]
                        sigma_values[edge_idx] = child_sum
                        total += child_sum
                subtree_sum[node] = total

        # Step 2: Compute result = Δσ (JIT-compiled)
        result = np.zeros(self.m, dtype=np.float64)

        if self.column_delta_indices is not None and self.column_edge_counts is not None and self.column_edge_counts.size:
            sigma_repeated = np.repeat(sigma_values, self.column_edge_counts)
            # Use JIT-compiled result accumulation
            _column_mst_result_jit(self.column_delta_indices, 
                                   self.column_delta_values, 
                                   sigma_repeated, 
                                   result)
        else:
            # Fallback path
            for idx, edge in enumerate(self.tree_column.edges):
                sigma_val = sigma_values[idx]
                for index, val in zip(edge.sparse_delta.nonzero_indices, edge.sparse_delta.nonzero_values):
                    result[index] += val * sigma_val

        return result.astype(v.dtype)
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        original_nnz = np.count_nonzero(self.original_matrix)
        tree = self.tree_column if self.use_column_mst else self.tree_row
        return {
            'matrix_shape': f"{self.m}×{self.n}",
            'original_nonzeros': original_nnz,
            'tree_weight': tree.weight,
            'compression_ratio': tree.weight / original_nnz if original_nnz > 0 else 0,
            'num_edges': len(tree.edges),
            'algorithm': 'MST(M) [column-wise]'
        }
