# Project Development Progress & Roadmap

## 📌 Architecture State
- **Framework**: PyTorch Geometric (PyG), PyTorch, NetworkX, SciPy
- **Dataset**: SNAP Facebook ego-networks (`data/ego-facebook/`)
- **Structure**:
  - `src/data/`: Dataset loaders, transforms, samplers
  - `src/models/`: GNN encoders (GCN, GAT, base interfaces), link predictors
  - `src/training/`: Training loop (`trainer.py`), early stopping, checkpointing
  - `src/evaluation/`: Baselines, ranking metrics
  - `src/utils/`: YAML configuration loader (`config.py`), logger
  - `configs/default.yaml`: Central hyperparameters

---

## 🚀 Step Progress

### ✅ Step 1: Modular Project Architecture & Base Pipeline
- [x] Standardized package layout (`src/data`, `src/models`, `src/training`, `src/evaluation`, `src/utils`)
- [x] Configuration management via `configs/default.yaml`
- [x] Reusable trainer with early stopping in `src/training/trainer.py`
- [x] Base entrypoint `train.py`

### ✅ Step 2: Classical Baselines, Ranking Metrics & Benchmark Runner
- [x] Classical link prediction heuristics (`src/evaluation/baselines.py`): Common Neighbors (CN), Adamic-Adar (AA), Jaccard Coefficient (JC), Preferential Attachment (PA), Resource Allocation (RA)
- [x] Recommendation ranking metrics (`src/evaluation/metrics.py`): Precision@K, Recall@K, NDCG@K, MRR (+ fixed missing `Callable` import for `bootstrap_confidence_interval`)
- [x] Multi-seed (5 seeds) & multi-split runner (`benchmark.py`) saving output to `results/benchmark_summary.md`

### 📋 Step 3: Anti-Over-Smoothing & Scalable Positional Encodings
- [ ] Structural positional encodings: RWPE (Random Walk), LapPE (Laplacian eigenvectors) in `src/data/transforms.py`
- [ ] Anti-over-smoothing architectures in `src/models/encoders.py`: Residual connections, LayerNorm/PairNorm, Jumping Knowledge (JK-Net)
- [ ] Depth ablation (`depth_ablation.py`) & Dirichlet energy computation

### 📋 Step 4: Multi-Dataset Ingestion & Mini-Batch Sampling
- [ ] Loaders for SNAP Twitter, SNAP Google+, and OGB (`ogbl-collab`)
- [ ] Scalable mini-batch training with `LinkNeighborLoader` in `src/training/minibatch.py`

### 📋 Step 5: Graph Contrastive Learning (GCL) & Cold-Start Analysis
- [ ] Structural & feature augmentations (SGL, SimGCL) in `src/models/augmentations.py`
- [ ] InfoNCE loss & joint objective in `src/training/losses.py`
- [ ] Low-degree cold-start evaluation in `experiments/cold_start_eval.py`
