"""Construção de grafos de similaridade com média ponderada por atributo."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def _text_similarity_matrix(texts1, texts2, similarity_func: str) -> np.ndarray:
    """Return a pairwise similarity matrix between two text lists."""

    if similarity_func == "cosine":
        vectorizer = TfidfVectorizer().fit(texts1 + texts2)
        tfidf1 = vectorizer.transform(texts1)
        tfidf2 = vectorizer.transform(texts2)
        return cosine_similarity(tfidf1, tfidf2)

    if similarity_func == "jaccard":
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

        return np.array([[jaccard(a, b) for b in texts2] for a in texts1])

    raise ValueError("similarity_func deve ser 'cosine' ou 'jaccard'")


def _numeric_similarity_matrix(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
    """Compute similarity matrix for numeric attributes (e.g., price)."""

    v1 = values1.astype(float)
    v2 = values2.astype(float)

    diff = np.abs(v1[:, None] - v2[None, :])
    max_val = np.maximum(v1[:, None], v2[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = 1 - np.divide(diff, max_val, out=np.zeros_like(diff), where=max_val != 0)
    sim[max_val == 0] = 0
    return np.clip(sim, 0, 1)


def build_similarity_graph(df1, df2, similarity_func: str = "cosine", weights=None):
    """Constroi o grafo bipartido G=(V1,V2,E) a partir de dois DataFrames.

    Cada atributo é comparado individualmente e a similaridade final é a média
    ponderada dessas similaridades. A coluna de título/nome recebe maior peso.

    Args:
        df1 (pd.DataFrame): registros da primeira base (V1)
        df2 (pd.DataFrame): registros da segunda base (V2)
        similarity_func (str): "cosine" ou "jaccard" para atributos textuais
        weights (dict, opcional): pesos para cada atributo

    Returns:
        tuple: (V1, V2, E)
    """

    # Pesos padrão priorizando o nome/título
    weights = weights or {
        "title": 0.5,
        "description": 0.2,
        "manufacturer": 0.2,
        "price": 0.1,
    }

    cols_text = [c for c in ["title", "description", "manufacturer"] if c in df1.columns and c in df2.columns]

    matrices = []
    total_w = 0.0

    for col in cols_text:
        w = weights.get(col, 0)
        if w <= 0:
            continue
        texts1 = df1[col].fillna("").astype(str).tolist()
        texts2 = df2[col].fillna("").astype(str).tolist()
        matrices.append(w * _text_similarity_matrix(texts1, texts2, similarity_func))
        total_w += w

    if "price" in df1.columns and "price" in df2.columns:
        w = weights.get("price", 0)
        if w > 0:
            parse_price = lambda s: float(re.sub(r"[^0-9.]", "", str(s)) or 0)
            prices1 = df1["price"].map(parse_price).to_numpy(dtype=float)
            prices2 = df2["price"].map(parse_price).to_numpy(dtype=float)
            matrices.append(w * _numeric_similarity_matrix(prices1, prices2))
            total_w += w

    if total_w == 0:
        sim_matrix = np.zeros((len(df1), len(df2)))
    else:
        sim_matrix = sum(matrices) / total_w

    V1 = set(df1["id"].astype(str))
    V2 = set(df2["id"].astype(str))
    E = []

    ids1 = df1["id"].astype(str).tolist()
    ids2 = df2["id"].astype(str).tolist()
    for i, vi in enumerate(ids1):
        for j, vj in enumerate(ids2):
            E.append((vi, vj, float(sim_matrix[i, j])))

    return V1, V2, E


__all__ = ["build_similarity_graph"]

