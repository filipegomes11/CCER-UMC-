"""Funções utilitárias para o pipeline de resolução de entidades."""

import pandas as pd


def print_results(clusters, unmatched_V1, unmatched_V2):
    """Imprime clusters e elementos não pareados."""

    print("Clusters encontrados:")
    for c in clusters:
        print(c)

    print("\nNão casados em V1:", unmatched_V1)
    print("Não casados em V2:", unmatched_V2)


def export_clusters(clusters, path):
    """Exporta a lista de clusters para um arquivo CSV."""

    df = pd.DataFrame(clusters, columns=["id_1", "id_2", "similarity"])
    df.to_csv(path, index=False)


def evaluate_clusters(clusters, ground_truth):
    """Calcula métricas de qualidade dos clusters usando o mapeamento limpo.

    Args:
        clusters (list[tuple[str, str, float]]): pares encontrados
        ground_truth (set[tuple[str, str]]): pares corretos

    Returns:
        dict: métricas calculadas
    """

    predicted = {(a, b) for a, b, _ in clusters}
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    matched_v1 = {a for a, _, _ in clusters}
    matched_v2 = {b for _, b, _ in clusters}
    gt_v1 = {a for a, _ in ground_truth}
    gt_v2 = {b for _, b in ground_truth}
    parity1 = len(matched_v1 & gt_v1) / len(gt_v1) if gt_v1 else 0.0
    parity2 = len(matched_v2 & gt_v2) / len(gt_v2) if gt_v2 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parity_dataset1": parity1,
        "parity_dataset2": parity2,
    }

