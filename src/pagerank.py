import numpy as np
import scipy.sparse as sp
import time
import io
import zipfile

class PageRankEngine:
    """
    A scalable PageRank engine designed to compute node importance in large directed graphs.
    Supports both iterative (Power Method) and analytical (Matrix Inversion) solvers.
    """
    def __init__(self, teleport_prob=0.15):
        """
        Initializes the PageRank engine.
        
        Args:
            teleport_prob (float): The probability 'p' of a random surfer teleporting 
                                   to a random page instead of following a link. 
                                   (Equivalent to 1 - beta in the lecture notes).
        """
        self.p = teleport_prob
        self.N = 0
        self.M = None
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.dead_ends = []

    def load_from_file(self, filepath):
        """
        Reads a standard edge-list text file (like web-Google.txt) and builds the graph.
        Automatically handles .zip archives and bypasses Windows encoding errors.
        """
        print(f"Loading graph from {filepath}...")
        edges = []
        
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as z:
                # Find the first .txt file inside the zip archive
                txt_files = [name for name in z.namelist() if name.endswith('.txt')]
                if not txt_files:
                    raise FileNotFoundError("No .txt file found inside the zip archive.")
                
                target_file = txt_files[0]
                print(f"Reading directly from {target_file} inside the archive...")
                
                with z.open(target_file) as f:
                    # Wrap the binary stream in a text reader, forcing UTF-8 and ignoring bad bytes
                    with io.TextIOWrapper(f, encoding='utf-8', errors='ignore') as text_file:
                        for line in text_file:
                            if line.startswith('#'):
                                continue
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                edges.append((parts[0], parts[1]))
        else:
            # Handle standard uncompressed text files with the same safety nets
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
                        
        self.fit_from_edges(edges)

    def fit_from_edges(self, edges):
        """
        Builds the column-stochastic transition matrix M from a list of directed edges.
        This modularity allows it to be reused later for the GraphRAG pipeline.
        
        Args:
            edges (list of tuples): Directed edges e.g., [("NodeA", "NodeB"), ...]
        """
        print("Extracting nodes and building index mappings...")
        # Extract unique nodes and create bidirectional mappings (handles non-contiguous IDs)
        nodes = set([n for edge in edges for n in edge])
        self.N = len(nodes)
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}

        print(f"Processing {self.N} unique nodes and {len(edges)} edges...")

        # Calculate out-degrees to normalize the transition probabilities
        out_degree = np.zeros(self.N)
        for src, dst in edges:
            out_degree[self.node_to_idx[src]] += 1

        # Identify dead ends (nodes with 0 out-links) to handle rank leakage
        self.dead_ends = np.where(out_degree == 0)[0]

        # Construct the Sparse Column-Stochastic Matrix M
        # M_ij = 1 / out_degree[j] if there is a link from j to i
        row, col, data = [], [], []
        for src, dst in edges:
            u = self.node_to_idx[src] # Column (Source)
            v = self.node_to_idx[dst] # Row (Destination)
            
            row.append(v)
            col.append(u)
            data.append(1.0 / out_degree[u]) 

        # Using csc_matrix (Compressed Sparse Column) as it is efficient for column operations
        self.M = sp.csc_matrix((data, (row, col)), shape=(self.N, self.N))
        print("Sparse transition matrix M successfully built.\n")

    def solve_iterative(self, max_iter=100, tol=1e-6):
        """
        Calculates PageRank using the scalable Power Method.
        
        Args:
            max_iter (int): Maximum number of iterations before forcing a stop.
            tol (float): Tolerance threshold for convergence.
            
        Returns:
            np.ndarray: The final PageRank vector R.
        """
        print("Starting Iterative Solver (Power Method)...")
        start_time = time.time()
        
        # Initialize the PageRank vector R uniformly: 1/N for each node
        R = np.ones(self.N) / self.N 
        
        for i in range(max_iter):
            R_prev = R.copy()
            
            # Step 1: Distribute rank through explicit links
            # (1 - p) * M * R
            link_distribution = (1 - self.p) * self.M.dot(R)
            
            # Step 2: Handle Dead Ends mathematically
            # Surfers at dead ends jump to a random page with probability 1.
            dead_end_mass = np.sum(R[self.dead_ends])
            
            # Step 3: Combine teleportation probability (p) and dead-end mass
            # Both distribute mass uniformly across all N nodes.
            total_teleport_mass = (self.p + (1 - self.p) * dead_end_mass) / self.N
            
            # Update R
            R = link_distribution + total_teleport_mass
            
            # Step 4: Check for convergence (L1 norm of the difference)
            err = np.linalg.norm(R - R_prev, ord=1) 
            if err <= tol:
                print(f"-> Converged in {i+1} iterations ({time.time() - start_time:.4f} seconds).")
                return R
                
        print(f"-> Warning: Reached max iterations ({max_iter}) without strict convergence.")
        return R

    def solve_analytical(self):
        """
        Calculates the exact Closed-Form solution: R = (p/N) * (I - (1-p)M)^-1 * 1.
        WARNING: Converts matrix to dense format. Will cause MemoryError if N > ~15,000.
        
        Returns:
            np.ndarray: The exact analytical PageRank vector R.
        """
        if self.N > 20000:
            raise MemoryError(f"Graph too large ({self.N} nodes) for dense analytical inversion. Use solve_iterative() instead.")
            
        print("Starting Analytical Solver (Matrix Inversion)...")
        start_time = time.time()
        
        # Convert sparse matrix M to dense format
        M_dense = self.M.toarray()
        
        # Apply the dead-end correction directly to the matrix:
        # A dead-end column becomes a uniform probability distribution (1/N)
        for dead_idx in self.dead_ends:
            M_dense[:, dead_idx] = 1.0 / self.N
            
        # Create Identity Matrix I
        I_dense = np.eye(self.N)
        
        # Formulate A = I - (1-p)M
        A_dense = I_dense - (1 - self.p) * M_dense
        
        # Formulate b = (p/N) * 1
        b_vector = np.ones(self.N) * (self.p / self.N)
        
        # Solve the linear system A * R = b
        R = np.linalg.solve(A_dense, b_vector)
        
        print(f"-> Analytical Matrix Inversion finished in {time.time() - start_time:.4f} seconds.")
        return R

    def solve_personalized(self, seed_nodes, max_iter=100, tol=1e-6):
        """
        Calculates Personalized PageRank (PPR) biased toward seed nodes.
        
        Instead of uniform teleportation, the random surfer preferentially 
        returns to one of the given seed nodes. This amplifies the importance 
        of nodes close to the seeds in the graph structure.
        
        Args:
            seed_nodes (list): List of node names to personalize toward.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
            
        Returns:
            np.ndarray: The personalized PageRank vector R.
        """
        print(f"Starting Personalized PageRank Solver (biased toward {len(seed_nodes)} seeds)...")
        start_time = time.time()
        
        # Create personalization vector: 1/|seeds| for seed nodes, 0 for others
        personalization = np.zeros(self.N)
        seed_indices = []
        for seed in seed_nodes:
            if seed in self.node_to_idx:
                idx = self.node_to_idx[seed]
                seed_indices.append(idx)
                personalization[idx] = 1.0 / len(seed_nodes)
        
        if not seed_indices:
            print("No valid seeds found. Falling back to global PageRank.")
            return self.solve_iterative(max_iter=max_iter, tol=tol)
        
        # Initialize PageRank uniformly
        R = np.ones(self.N) / self.N
        
        for i in range(max_iter):
            R_prev = R.copy()
            
            # Step 1: Distribute rank through links
            link_distribution = (1 - self.p) * self.M.dot(R)
            
            # Step 2: Handle dead ends
            dead_end_mass = np.sum(R[self.dead_ends])
            
            # Step 3: Combine teleportation (to personalization vector) + dead-end mass
            # p * personalization: teleport to seed nodes
            # (1-p) * dead_end_mass / |seeds|: dead-end surfers redistribute to seeds
            teleport_mass = (self.p * personalization) + (
                ((1 - self.p) * dead_end_mass) / len(seed_indices) * 
                np.array([1.0 if idx in seed_indices else 0.0 for idx in range(self.N)])
            )
            
            # Update R
            R = link_distribution + teleport_mass
            
            # Step 4: Check convergence
            err = np.linalg.norm(R - R_prev, ord=1)
            if err <= tol:
                print(f"-> PPR Converged in {i+1} iterations ({time.time() - start_time:.4f} seconds).")
                return R
        
        print(f"-> PPR Warning: Reached max iterations ({max_iter}) without strict convergence.")
        return R

    def get_top_k(self, R, k=10):
        """
        Helper method to extract the top-ranked nodes.
        
        Args:
            R (np.ndarray): The calculated PageRank vector.
            k (int): Number of top nodes to return.
            
        Returns:
            list of tuples: [(NodeID, PageRankScore), ...]
        """
        # Get indices of the top k scores, sorted descending
        top_indices = np.argsort(R)[::-1][:k]
        return [(self.idx_to_node[i], R[i]) for i in top_indices]