import re
import numpy as np
from src.pagerank import PageRankEngine


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
    "what", "which", "who", "whom", "whose", "led", "later",
}


class GraphRAGRetriever:
    """
    Executes the GraphRAG retrieval pipeline.
    Uses PageRank to prioritize which facts from the Knowledge Graph
    should be returned as top-k retrieval context.

    Note:
        This mock project intentionally stops at retrieval output and
        does not include LLM answer generation.
    """

    def __init__(
        self,
        edges,
        teleport_prob=0.15,
        rerank_alpha=0.0,
        max_hops=2,
        bridge_weight=0.2,
        use_undirected_expansion=True,
        include_seed_nodes=False,
        seed_top_k=3,
    ):
        """
        Initializes the retriever and precomputes the PageRank for the entire Knowledge Graph.

        Args:
            edges (list of tuples): The extracted knowledge graph edges.
            teleport_prob (float): PageRank teleportation probability p.
            rerank_alpha (float): Legacy parameter (unused; kept for compatibility).
            max_hops (int): Number of hops for multi-hop expansion.
            bridge_weight (float): Weight for graph-bridge proximity in reranking.
            use_undirected_expansion (bool): Whether traversal can follow reverse edges.
            include_seed_nodes (bool): If False, seed nodes are excluded from final top-k.
            seed_top_k (int): Maximum number of seed nodes selected by vector search.
        """
        print("Initializing GraphRAG Retriever...")
        self.edges = edges
        self.rerank_alpha = 0.0
        self.max_hops = max_hops
        self.bridge_weight = bridge_weight
        self.use_undirected_expansion = use_undirected_expansion
        self.include_seed_nodes = include_seed_nodes
        self.seed_top_k = max(1, int(seed_top_k))

        # Keep ranking weights valid and deterministic.
        self.bridge_weight = min(max(self.bridge_weight, 0.0), 1.0)
        self.authority_weight = 1.0 - self.bridge_weight

        # 1. Initialize and fit the PageRank Engine we built earlier
        self.pr_engine = PageRankEngine(teleport_prob=teleport_prob)
        self.pr_engine.fit_from_edges(self.edges)

        # 2. Precompute the global authority of all concepts in the graph
        self.pagerank_vector = self.pr_engine.solve_iterative()

        # 3. Create a quick-lookup dictionary for Node -> PageRank Score
        self.node_scores = {
            node: self.pagerank_vector[idx]
            for node, idx in self.pr_engine.node_to_idx.items()
        }

        # Normalize PageRank for blending with bridge scores during reranking.
        pr_values = np.array(list(self.node_scores.values()), dtype=float)
        pr_min, pr_max = float(np.min(pr_values)), float(np.max(pr_values))
        if np.isclose(pr_max, pr_min):
            self.normalized_node_scores = {node: 1.0 for node in self.node_scores}
        else:
            self.normalized_node_scores = {
                node: (score - pr_min) / (pr_max - pr_min)
                for node, score in self.node_scores.items()
            }

        # Build an adjacency list for fast neighbor lookups during retrieval
        self.adjacency_list = {}
        for src, dst in self.edges:
            if src not in self.adjacency_list:
                self.adjacency_list[src] = []
            if dst not in self.adjacency_list:
                self.adjacency_list[dst] = []
            self.adjacency_list[src].append(dst)
            if self.use_undirected_expansion:
                self.adjacency_list[dst].append(src)

        # Build node text vectors for seed identification and query relevance.
        self.nodes = list(self.node_scores.keys())
        self._build_vector_index()

    def _tokenize(self, text):
        """Simple tokenizer for lightweight vector search in this mock project."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def _build_vector_index(self):
        """Builds TF-IDF vectors over node names for vector-based retrieval."""
        tokenized_nodes = [self._tokenize(node) for node in self.nodes]
        vocab_set = sorted({token for tokens in tokenized_nodes for token in tokens})
        self.vocab = {token: idx for idx, token in enumerate(vocab_set)}

        self.node_vector_matrix = np.zeros((len(self.nodes), len(self.vocab)), dtype=float)
        if len(self.vocab) == 0:
            self.idf = np.array([])
            self.node_to_row = {node: idx for idx, node in enumerate(self.nodes)}
            return

        doc_freq = np.zeros(len(self.vocab), dtype=float)
        for tokens in tokenized_nodes:
            for token in set(tokens):
                doc_freq[self.vocab[token]] += 1.0

        n_docs = max(len(self.nodes), 1)
        self.idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0

        for row_idx, tokens in enumerate(tokenized_nodes):
            if not tokens:
                continue

            tf = np.zeros(len(self.vocab), dtype=float)
            for token in tokens:
                tf[self.vocab[token]] += 1.0
            tf /= len(tokens)

            vec = tf * self.idf
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.node_vector_matrix[row_idx] = vec

        self.node_to_row = {node: idx for idx, node in enumerate(self.nodes)}

    def _encode_query(self, query):
        """Encodes query text into the same TF-IDF vector space as nodes."""
        if len(self.vocab) == 0:
            return np.array([])

        tf = np.zeros(len(self.vocab), dtype=float)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return tf

        for token in query_tokens:
            if token in self.vocab:
                tf[self.vocab[token]] += 1.0
        tf /= len(query_tokens)

        q_vec = tf * self.idf
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        return q_vec

    def _vector_seed_search(self, query, top_k=3):
        """Stage A: vector search to identify seed nodes from the query."""
        q_vec = self._encode_query(query)
        if q_vec.size == 0 or np.linalg.norm(q_vec) == 0:
            return []

        scores = self.node_vector_matrix.dot(q_vec)
        ranked_idx = np.argsort(scores)[::-1]

        seeds = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            seeds.append((self.nodes[idx], score))
            if len(seeds) >= top_k:
                break

        return seeds

    def _extract_seeds(self, query):
        """
        Identifies seed entities using vector similarity over node names.
        """
        return [node for node, _ in self._vector_seed_search(query, top_k=self.seed_top_k)]

    def _bfs_distances(self, start):
        """Unweighted shortest-path distances from a start node."""
        distances = {start: 0}
        frontier = [start]

        while frontier:
            next_frontier = []
            for node in frontier:
                for neighbor in self.adjacency_list.get(node, []):
                    if neighbor in distances:
                        continue
                    distances[neighbor] = distances[node] + 1
                    next_frontier.append(neighbor)
            frontier = next_frontier

        return distances

    def _compute_shortest_path_nodes(self, seed_a, seed_b):
        """Nodes that lie on at least one shortest path between two seeds."""
        if seed_a == seed_b:
            return {seed_a}

        dist_from_a = self._bfs_distances(seed_a)
        dist_from_b = self._bfs_distances(seed_b)
        if seed_b not in dist_from_a:
            return set()

        shortest_len = dist_from_a[seed_b]
        path_nodes = set()
        for node in self.nodes:
            da = dist_from_a.get(node)
            db = dist_from_b.get(node)
            if da is None or db is None:
                continue
            if da + db == shortest_len:
                path_nodes.add(node)

        return path_nodes

    def _expand_candidates(self, seeds):
        """Stage B: gather candidates and record minimum hop distance from seeds."""
        min_hops = {seed: 0 for seed in seeds}
        frontier = set(seeds)

        for depth in range(1, max(self.max_hops, 0) + 1):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency_list.get(node, []):
                    if neighbor not in min_hops:
                        min_hops[neighbor] = depth
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return min_hops

    def _bridge_score(self, node, seeds, seed_distances, path_nodes):
        """Connector score using multi-seed proximity plus shortest-path membership."""
        if node in seeds:
            return 0.0

        distances = []
        for seed in seeds:
            d = seed_distances.get(seed, {}).get(node)
            if d is not None and d > 0:
                distances.append(d)

        if not distances:
            return 0.0

        coverage = len(distances) / max(len(seeds), 1)
        closeness = len(distances) / float(sum(distances))
        path_bonus = 1.0 if node in path_nodes else 0.0

        return 0.6 * (coverage * closeness) + 0.4 * path_bonus

    def _rerank_nodes(self, candidate_hops, seeds, seed_distances, path_nodes):
        """Stage C: rerank candidates by blending authority and bridge score."""
        reranked = []

        for node in candidate_hops:
            authority = self.normalized_node_scores[node]
            bridge = self._bridge_score(node, seeds, seed_distances, path_nodes)
            final_score = (
                self.authority_weight * authority
                + self.bridge_weight * bridge
            )
            reranked.append((node, final_score, authority, bridge, candidate_hops[node]))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def retrieve_context(self, query, top_k=5):
        """
        Full retrieval pipeline (vector seeds -> expansion -> reranking).

        Args:
            query (str): The user's question.
            top_k (int): How many high-authority nodes to retrieve.

        Returns:
            list: The most authoritative entities related to the query.
        """
        print(f"\n[Retrieval] Analyzing Query: '{query}'")

        # Step 1: Find entry points (Seeds) with vector similarity
        seed_pairs = self._vector_seed_search(query, top_k=self.seed_top_k)
        seeds = [node for node, _ in seed_pairs]
        print(f"[Retrieval] Identified Seeds: {seeds}")

        if not seeds:
            return []

        # Step 2: Gather multi-hop neighbors from seeds
        candidate_hops = self._expand_candidates(seeds)
        seed_distances = {seed: self._bfs_distances(seed) for seed in seeds}
        path_nodes = set()
        if len(seeds) >= 2:
            path_nodes = self._compute_shortest_path_nodes(seeds[0], seeds[1])

        # Step 3: Rerank by combined PageRank authority + bridge.
        ranked_nodes = self._rerank_nodes(candidate_hops, seeds, seed_distances, path_nodes)
        if not self.include_seed_nodes:
            ranked_nodes = [row for row in ranked_nodes if row[0] not in seeds]

        # Extract just the node names for the top_k results
        top_context_nodes = [node for node, _, _, _, _ in ranked_nodes[:top_k]]
        print(f"[Retrieval] Selected Top {top_k} Nodes: {top_context_nodes}")

        return top_context_nodes

    def retrieve_context_with_facts(self, query, top_k=5):
        """
        Returns top-k retrieved nodes using GLOBAL PageRank.

        This method is intentionally LLM-free for assignment scope.

        Args:
            query (str): The user's question.
            top_k (int): Number of nodes to return.

        Returns:
            dict: {
                "query": str,
                "pagerank_type": "global",
                "seeds": list[str],
                "top_nodes": list[str],
                "context_facts": list[str],
                "scores": list[dict]
            }
        """
        seed_pairs = self._vector_seed_search(query, top_k=self.seed_top_k)
        seeds = [node for node, _ in seed_pairs]
        if not seeds:
            return {
                "query": query,
                "pagerank_type": "global",
                "seeds": [],
                "top_nodes": [],
                "context_facts": [],
                "scores": []
            }

        candidate_hops = self._expand_candidates(seeds)
        seed_distances = {seed: self._bfs_distances(seed) for seed in seeds}
        path_nodes = set()
        if len(seeds) >= 2:
            path_nodes = self._compute_shortest_path_nodes(seeds[0], seeds[1])

        ranked_nodes = self._rerank_nodes(candidate_hops, seeds, seed_distances, path_nodes)
        if not self.include_seed_nodes:
            ranked_nodes = [row for row in ranked_nodes if row[0] not in seeds]

        top_ranked = ranked_nodes[:top_k]
        context_nodes = [node for node, _, _, _, _ in top_ranked]

        # Retrieve the specific edges (facts) that connect these top nodes
        context_facts = []
        for src, dst in self.edges:
            if src in context_nodes and dst in context_nodes:
                context_facts.append(f"{src} -> {dst}")

        return {
            "query": query,
            "pagerank_type": "global",
            "seeds": seeds,
            "top_nodes": context_nodes,
            "context_facts": context_facts,
            "scores": [
                {
                    "node": node,
                    "final": final_score,
                    "pagerank": authority,
                    "bridge": bridge,
                    "hop": hop,
                }
                for node, final_score, authority, bridge, hop in top_ranked
            ]
        }

    def retrieve_context_with_personalized_pagerank(self, query, top_k=5):
        """
        Returns top-k retrieved nodes using Personalized PageRank (PPR).
        
        PPR biases the PageRank computation toward seed nodes identified by vector search.
        This makes query-relevant entities and their direct neighbors rank higher.
        
        Args:
            query (str): The user's question.
            top_k (int): Number of nodes to return.
            
        Returns:
            dict: {
                "query": str,
                "pagerank_type": "personalized",
                "seeds": list[str],
                "top_nodes": list[str],
                "context_facts": list[str],
                "scores": list[dict]
            }
        """
        # Identify seeds via vector search
        seed_pairs = self._vector_seed_search(query, top_k=self.seed_top_k)
        seeds = [node for node, _ in seed_pairs]
        if not seeds:
            return {
                "query": query,
                "pagerank_type": "personalized",
                "seeds": [],
                "top_nodes": [],
                "context_facts": [],
                "scores": []
            }

        # Compute Personalized PageRank biased toward seeds
        ppr_vector = self.pr_engine.solve_personalized(seeds)
        ppr_node_scores = {
            node: ppr_vector[idx]
            for node, idx in self.pr_engine.node_to_idx.items()
        }
        
        # Normalize PPR scores for blending
        ppr_values = np.array(list(ppr_node_scores.values()), dtype=float)
        ppr_min, ppr_max = float(np.min(ppr_values)), float(np.max(ppr_values))
        if np.isclose(ppr_max, ppr_min):
            normalized_ppr_scores = {node: 1.0 for node in ppr_node_scores}
        else:
            normalized_ppr_scores = {
                node: (score - ppr_min) / (ppr_max - ppr_min)
                for node, score in ppr_node_scores.items()
            }

        # Expand candidates and compute distances
        candidate_hops = self._expand_candidates(seeds)
        seed_distances = {seed: self._bfs_distances(seed) for seed in seeds}
        path_nodes = set()
        if len(seeds) >= 2:
            path_nodes = self._compute_shortest_path_nodes(seeds[0], seeds[1])

        # Rerank using PPR instead of global PageRank
        reranked = []

        for node in candidate_hops:
            authority = normalized_ppr_scores[node]  # Use PPR instead of global PageRank
            bridge = self._bridge_score(node, seeds, seed_distances, path_nodes)
            final_score = (
                self.authority_weight * authority
                + self.bridge_weight * bridge
            )
            reranked.append((node, final_score, authority, bridge, candidate_hops[node]))

        reranked.sort(key=lambda x: x[1], reverse=True)
        if not self.include_seed_nodes:
            reranked = [row for row in reranked if row[0] not in seeds]

        top_ranked = reranked[:top_k]
        context_nodes = [node for node, _, _, _, _ in top_ranked]

        # Retrieve context facts connecting top nodes
        context_facts = []
        for src, dst in self.edges:
            if src in context_nodes and dst in context_nodes:
                context_facts.append(f"{src} -> {dst}")

        return {
            "query": query,
            "pagerank_type": "personalized",
            "seeds": seeds,
            "top_nodes": context_nodes,
            "context_facts": context_facts,
            "scores": [
                {
                    "node": node,
                    "final": final_score,
                    "pagerank": authority,
                    "bridge": bridge,
                    "hop": hop,
                }
                for node, final_score, authority, bridge, hop in top_ranked
            ]
        }
