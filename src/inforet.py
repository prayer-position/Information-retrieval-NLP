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
        print("Dow<nloading Stanford IMDb dataset...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    
    if not os.path.exists(DATA_DIR):
        print("Extracting dataset...")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall()

def load_split(split: str) -> Tuple[List[str], np.ndarray]:
    """
    Returns texts and labels for split in {"train", "test"}.
    label : 1=pos, 0=neg 
    """
    base = os.path.join(DATA_DIR, split)
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

def topk_indices(doc_matrix, query_matrix, k: int) -> np.ndarray:
    """
    Compute top-k doc indices for each query using cosine similarity. 
    With sklearn TF-YDF (L2-normalized), cosine similarity == dot product. 
    Returns shape: (n_queries, k)
    """
    sims = (doc_matrix @ query_matrix.T).toarray().T # (n_queries, n_docs)
    k = min(k, sims.shape[1])

    part = np.argpartition(sims, -k, axis=1)[:, -k] # (n_queris, k) unordered
    row = np.arange(sims.shape[0])[:, None]
    order = np.argsort(sims[row, part], axis = 1)[:, ::-1]
    
    return part[row, order]

def precision_at_k(retrieved_idx: np.ndarray, doc_labels: np.ndarray, query_labels: np.ndarray) -> float:
    """
    Meazn Precision@K across queries:
    fraction of top-K retrieved docs whose label mathces query label
    """
    k = retrieved_idx.shape[1]

    retrieved_labels = doc_labels[retrieved_idx] #(n_queries, k)
    matches = (retrieved_labels == query_labels[:, None]).mean(axis = 1) # per-query precision@K
    return float(matches.mean())