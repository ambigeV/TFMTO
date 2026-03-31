"""
TabPFN Embedding Extraction — Demo Snippets
============================================
Shows how to call model.get_embeddings() with TabPFN v2.5 in both
standalone and multi-task (TFM-pooled) settings.

Requires: tabpfn >= 6.x  (tested on 6.4.1)
"""
import sys
import numpy as np

sys.path.insert(0, '../src')

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from ddmtolab.Methods.Algo_Methods.tfm_utils import (
    append_task_id, append_task_id_onehot, pad_to_dim,
)


# ---------------------------------------------------------------------------
# Helper: build a TabPFN v2.5 regressor (same settings as TFM variants)
# ---------------------------------------------------------------------------
def build_model(n_estimators: int = 8, random_state: int = 42) -> TabPFNRegressor:
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
    model.set_params(
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
    )
    return model


# ---------------------------------------------------------------------------
# Snippet 1 — Basic embedding extraction
# ---------------------------------------------------------------------------
def snippet_basic():
    """Fit on random data, extract test and train embeddings."""
    rng = np.random.default_rng(0)
    X_train = rng.random((40, 5))
    y_train = np.sin(X_train[:, 0]) + rng.normal(0, 0.05, 40)
    X_test  = rng.random((10, 5))

    model = build_model(n_estimators=4)
    model.fit(X_train, y_train)

    # Test-token embeddings  → shape (n_estimators, n_test,  embed_dim)
    emb_test  = model.get_embeddings(X_test,  data_source='test')
    # Train-token embeddings → shape (n_estimators, n_train, embed_dim)
    emb_train = model.get_embeddings(X_train, data_source='train')

    print(f"[basic] emb_test  shape: {emb_test.shape}")   # (4, 10, D)
    print(f"[basic] emb_train shape: {emb_train.shape}")  # (4, 40, D)

    # --- Reduce across ensemble axis ---
    emb_avg = emb_test.mean(axis=0)               # average  → (n_test, D)
    emb_cat = emb_test.reshape(emb_test.shape[1], -1)  # concat → (n_test, 4*D)

    print(f"[basic] emb_avg  shape: {emb_avg.shape}")
    print(f"[basic] emb_cat  shape: {emb_cat.shape}")
    return emb_avg


# ---------------------------------------------------------------------------
# Snippet 2 — Multi-task pooled embeddings (scalar task ID, TFM-Uni style)
# ---------------------------------------------------------------------------
def snippet_multitask_scalar(n_tasks: int = 3, dim: int = 5, n_train: int = 30):
    """
    Replicates the MTBO-TFM-Uni data layout:
      feature layout: [x_0, ..., x_{max_dim-1}, task_id]
    Embeddings capture cross-task context because all tasks share the
    same transformer context window.
    """
    rng = np.random.default_rng(1)
    decs = [rng.random((n_train, dim)) for _ in range(n_tasks)]
    objs = [np.sin(decs[i][:, 0]) + rng.normal(0, 0.05, n_train) for i in range(n_tasks)]

    # Build pooled dataset (same as MTBO_TFM_Uniform._build_pooled_dataset)
    X_parts, y_parts = [], []
    for j in range(n_tasks):
        X_j = append_task_id(decs[j], j)          # [x..., j]
        y_j = decs[j][:, 0]                        # simplified (no normalise here)
        X_parts.append(X_j)
        y_parts.append(y_j)
    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    model = build_model(n_estimators=4)
    model.fit(X_all, y_all)

    # Extract embeddings for task-0 candidates
    task_id = 0
    candidates = rng.random((20, dim))
    C_enc = append_task_id(candidates, task_id)    # encode as task 0

    emb = model.get_embeddings(C_enc, data_source='test')  # (n_est, 20, D)
    emb_avg = emb.mean(axis=0)                             # (20, D)

    print(f"[multitask-scalar] emb shape (raw):  {emb.shape}")
    print(f"[multitask-scalar] emb shape (avg):  {emb_avg.shape}")
    return emb_avg


# ---------------------------------------------------------------------------
# Snippet 3 — Multi-task pooled embeddings (one-hot task ID, TFM-Uni-OH style)
# ---------------------------------------------------------------------------
def snippet_multitask_onehot(n_tasks: int = 3, dim: int = 5, n_train: int = 30):
    """
    Replicates the MTBO-TFM-Uni-OH data layout:
      feature layout: [x_0, ..., x_{max_dim-1}, oh_0, ..., oh_{K-1}]
    One-hot encoding removes the false ordinal relationship between task IDs.
    """
    rng = np.random.default_rng(2)
    decs = [rng.random((n_train, dim)) for _ in range(n_tasks)]

    X_parts, y_parts = [], []
    for j in range(n_tasks):
        X_j = append_task_id_onehot(decs[j], j, n_tasks)
        y_j = decs[j][:, 0]
        X_parts.append(X_j)
        y_parts.append(y_j)
    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    model = build_model(n_estimators=4)
    model.fit(X_all, y_all)

    task_id = 1
    candidates = rng.random((20, dim))
    C_enc = append_task_id_onehot(candidates, task_id, n_tasks)

    emb = model.get_embeddings(C_enc, data_source='test')
    emb_avg = emb.mean(axis=0)

    print(f"[multitask-onehot] emb shape (raw):  {emb.shape}")
    print(f"[multitask-onehot] emb shape (avg):  {emb_avg.shape}")
    return emb_avg


# ---------------------------------------------------------------------------
# Snippet 4 — Train embeddings for task-similarity analysis
# ---------------------------------------------------------------------------
def snippet_task_similarity(n_tasks: int = 3, dim: int = 5, n_train: int = 30):
    """
    Use train-token embeddings to measure pairwise task similarity.
    Each task's centroid in embedding space is computed; cosine similarity
    between centroids reflects how the transformer relates the tasks.
    """
    rng = np.random.default_rng(3)
    decs = [rng.random((n_train, dim)) for _ in range(n_tasks)]

    X_parts, y_parts = [], []
    for j in range(n_tasks):
        X_j = append_task_id(decs[j], j)
        X_parts.append(X_j)
        y_parts.append(decs[j][:, 0])
    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    model = build_model(n_estimators=4)
    model.fit(X_all, y_all)

    # Train embeddings: (n_est, n_train_total, D)
    emb_train = model.get_embeddings(X_all, data_source='train').mean(axis=0)

    # Split back into per-task blocks and compute centroids
    centroids = []
    for j in range(n_tasks):
        block = emb_train[j * n_train : (j + 1) * n_train]  # (n_train, D)
        centroids.append(block.mean(axis=0))

    # Cosine similarity matrix
    sim = np.zeros((n_tasks, n_tasks))
    for a in range(n_tasks):
        for b in range(n_tasks):
            ca, cb = centroids[a], centroids[b]
            sim[a, b] = (ca @ cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)

    print("[task-similarity] cosine similarity matrix:")
    print(np.round(sim, 4))
    return sim


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== Snippet 1: Basic ===")
    snippet_basic()

    print("\n=== Snippet 2: Multi-task scalar ID ===")
    snippet_multitask_scalar()

    print("\n=== Snippet 3: Multi-task one-hot ID ===")
    snippet_multitask_onehot()

    print("\n=== Snippet 4: Task similarity via train embeddings ===")
    snippet_task_similarity()
