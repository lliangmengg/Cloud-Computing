class KnowledgeGraphBuilder:
    """
    Builds a Knowledge Graph as directed (source, target) edges.

    This assignment implementation is intentionally mock-first and starts from
    fixed knowledge triplets to simulate ingestion without an LLM layer.
    """
    def __init__(self, llm_client=None):
        """
        Initializes the builder.
        
        Args:
            llm_client: Retained for compatibility; not used in this assignment scope.
        """
        self.llm_client = llm_client
        self.edges = []
        self.nodes = set()

    def extract_edges_mock(self, query_context=None):
        """
        Deterministic ingestion from fixed triplets for the mock GraphRAG demo.

        Args:
            query_context (str, optional): Ignored; kept for backward compatibility.
        
        Returns:
            list of tuples: Representing directed edges [(Source, Target), ...]
        """
        print("Extracting entities and relationships (Mock Mode)...")
        
        # Simulated extraction from a document about Marie Curie
        mock_extracted_edges = [
            ("Marie Curie", "Radium"),
            ("Marie Curie", "Polonium"),
            ("Marie Curie", "Nobel Prize in Physics"),
            ("Marie Curie", "Nobel Prize in Chemistry"),
            ("Radium", "Radioactivity"),
            ("Polonium", "Radioactivity"),
            ("Radioactivity", "X-Rays"),
            ("X-Rays", "Radiotherapy"),
            ("Radiotherapy", "Medical Imaging"),
            ("Radiotherapy", "Cancer Treatment"),
            ("Medical Imaging", "Modern Diagnostics"),
            # Adding some noise (irrelevant facts) to prove PageRank filters them out
            ("Marie Curie", "Paris"),
            ("Paris", "Eiffel Tower"),
            ("Eiffel Tower", "Tourism")
        ]
        
        self._update_graph(mock_extracted_edges)
        return self.edges

    def ingest_triplets(self, triplets):
        """
        Ingests a user-provided list of triplets/edges directly.

        Args:
            triplets (list[tuple[str, str]]): Directed graph edges.

        Returns:
            list[tuple[str, str]]: Full graph edge list after update.
        """
        self._update_graph(triplets)
        return self.edges

    def extract_edges_llm(self, text):
        """
        Out-of-scope for this assignment build.
        """
        pass

    def _update_graph(self, new_edges):
        """Internal helper to maintain the list of edges and unique nodes."""
        self.edges.extend(new_edges)
        for src, dst in new_edges:
            self.nodes.add(src)
            self.nodes.add(dst)
        print(f"Graph updated: Currently holds {len(self.nodes)} nodes and {len(self.edges)} edges.")

    def get_edges(self):
        """Returns the full list of edges ready for the PageRankEngine."""
        return self.edges