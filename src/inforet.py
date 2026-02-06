import os 
import tarfile
import urllib.request
from typing import List, Tuple

from pathlib import Path
import numpy as np

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

DATA = Path.cwd().parent / "Data"
ARCHIVE = DATA / "aclImdb_v1.tar.gz"
DATA_DIR = DATA / "aclImdb"

def download_and_extract() -> None:
    if not os.path.exists(ARCHIVE):
        print("Downloading Stanford IMDb dataset...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    
    if not os.path.exists(DATA_DIR):
        print("Extracting dataset...")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(path = DATA)

def load_split(split: str) -> Tuple[List[str], np.ndarray]:
    """
    Returns texts and labels for split in {"train", "test"}.
    label : 1=pos, 0=neg 
    """
    base = DATA_DIR / split
    texts: List[str] = []
    labels: List[int] = []

    for label_name, y in (("pos", 1), ("neg", 0)):
        folder = os.path.join(base, label_name)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            with open(path, encoding = "utf-8") as f:
                texts.append(f.read())
            labels.append(y)


    return texts, np.array(labels, dtype=int)

def batched_topk_indices(doc_matrix, query_matrix, k: int, batch_size: int = 1000) -> np.ndarray:
    """
    Compute top-k indices using batching to save memory.
    doc_matrix: (n_docs, n_features) - sparse or dense
    query_matrix: (n_queries, n_features) - sparse or dense
    """
    n_queries = query_matrix.shape[0]
    n_docs = doc_matrix.shape[0]
    k = min(k, n_docs)
    
    all_topk_idx = np.zeros((n_queries, k), dtype=np.int32)
    doc_matrix_T = doc_matrix.T

    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        
        # Result shape: (batch_size, n_docs)
        batch_sims = query_matrix[start_idx:end_idx].dot(doc_matrix_T).toarray()
        
        # Find top-k in this batch (partition + sort)
        current_batch_size = end_idx - start_idx
        unsorted_topk = np.argpartition(batch_sims, -k, axis=1)[:, -k:]
        
        row_idx = np.arange(current_batch_size)[:, None]
        topk_sims = batch_sims[row_idx, unsorted_topk]
        sorted_within_topk = np.argsort(topk_sims, axis=1)[:, ::-1]
        
        # Store result in the pre-allocated array
        all_topk_idx[start_idx:end_idx] = unsorted_topk[row_idx, sorted_within_topk]
        
    return all_topk_idx

def precision_at_k(retrieved_idx: np.ndarray, doc_labels: np.ndarray, query_labels: np.ndarray) -> float:
    """
    Meazn Precision@K across queries:
    fraction of top-K retrieved docs whose label mathces query label
    """
    k = retrieved_idx.shape[1]

    retrieved_labels = doc_labels[retrieved_idx] #(n_queries, k)
    matches = (retrieved_labels == query_labels[:, None]).mean(axis = 1) # per-query precision@K
    return float(matches.mean())