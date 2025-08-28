# similarity.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_graph(dataset1, dataset2, similarity_func="cosine"):
    """Constroi o grafo bipartido G=(V1,V2,E) a partir de duas listas de
    registros.

    Os datasets devem ser listas de tuplas ``(id, texto)`` onde ``id`` é o
    identificador do registro e ``texto`` é a string utilizada no cálculo da
    similaridade.

    Args:
        dataset1 (list[tuple[str, str]]): registros da primeira base (V1)
        dataset2 (list[tuple[str, str]]): registros da segunda base (V2)
        similarity_func (str): "cosine" ou "jaccard"

    Returns:
        tuple: (V1, V2, E)
    """

    V1 = {vid for vid, _ in dataset1}
    V2 = {vid for vid, _ in dataset2}
    texts1 = [text for _, text in dataset1]
    texts2 = [text for _, text in dataset2]
    E = []

    if similarity_func == "cosine":
        vectorizer = TfidfVectorizer().fit(texts1 + texts2)
        tfidf1 = vectorizer.transform(texts1)
        tfidf2 = vectorizer.transform(texts2)
        sim_matrix = cosine_similarity(tfidf1, tfidf2)
    elif similarity_func == "jaccard":
        def jaccard(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

        sim_matrix = np.array([[jaccard(a, b) for b in texts2] for a in texts1])
    else:
        raise ValueError("similarity_func deve ser 'cosine' ou 'jaccard'")

    for i, (vi, _) in enumerate(dataset1):
        for j, (vj, _) in enumerate(dataset2):
            sim = float(sim_matrix[i, j])
            E.append((vi, vj, sim))

    return V1, V2, E

