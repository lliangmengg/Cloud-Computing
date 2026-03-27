# Cloud Computing Assignment: PageRank and GraphRAG

This repository contains two main parts:

1. PageRank math + implementation + validation on a web graph dataset.
2. A GraphRAG-style retrieval demo that uses PageRank scores for multi-hop context ranking.

## File-by-File Guide

### Root Files
- [README.md](README.md): Project overview and usage notes (this file).

### Data
- [data](data): Data directory.
- [data/raw](data/raw): Raw input datasets.
- [data/raw/web-Google_10k.zip](data/raw/web-Google_10k.zip): Google web graph subset used for PageRank experiments.

### Notebooks
- [notebooks](notebooks): Jupyter notebooks for experiments and demonstrations.
- [notebooks/01_pagerank_math_and_experiments.ipynb](notebooks/01_pagerank_math_and_experiments.ipynb):  
  Derives and validates PageRank (iterative vs analytical), then analyzes sensitivity to teleport probability \(p\).
- [notebooks/02_graphrag.ipynb](notebooks/02_graphrag.ipynb):  
  Runs mock GraphRAG retrieval for a multi-hop query and compares global vs personalized PageRank ranking.

### Source Code
- [src](src): Python package containing core implementation.
- [src/pagerank.py](src/pagerank.py): PageRank engine implementation (iterative, analytical, and personalized variants).
- [src/graph_builder.py](src/graph_builder.py): Knowledge graph construction utilities (mock extraction and triplet ingestion).
- [src/graph_retriever.py](src/graph_retriever.py): GraphRAG-style retrieval pipeline (seed selection, graph expansion, reranking, context facts).
- [src/__pycache__](src/__pycache__): Python bytecode cache generated automatically.

## Quick Workflow
1. Run [notebooks/01_pagerank_math_and_experiments.ipynb](notebooks/01_pagerank_math_and_experiments.ipynb) for PageRank derivation and validation.
2. Run [notebooks/02_graphrag.ipynb](notebooks/02_graphrag.ipynb) for GraphRAG retrieval experiments.
