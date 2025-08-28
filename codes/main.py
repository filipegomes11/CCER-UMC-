# main.py
import pandas as pd
from pathlib import Path

from similarity import build_similarity_graph
from umc import unique_mapping_clustering
from utils import export_clusters, evaluate_clusters
from cleanner import clean_all_csvs


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    datasets_dir = (base_dir / ".." / "datasets" / "Amazon-GoogleProducts").resolve()
    cleaned_dir = (base_dir / ".." / "cleaned_datasets" / "Amazon-GoogleProducts").resolve()

    if not cleaned_dir.exists():
        cleaned_dir.mkdir(parents=True)

    required_files = [
        "cleaned_Amazon.csv",
        "cleaned_GoogleProducts.csv",
        "cleaned_Amzon_GoogleProducts_perfectMapping.csv",
    ]
    if not all((cleaned_dir / f).exists() for f in required_files):
        clean_all_csvs(str(datasets_dir), str(cleaned_dir))

    amazon_path = cleaned_dir / "cleaned_Amazon.csv"
    google_path = cleaned_dir / "cleaned_GoogleProducts.csv"

    df_amazon = pd.read_csv(amazon_path, encoding="utf-8")
    df_google = pd.read_csv(google_path, encoding="utf-8").rename(columns={"name": "title"})

    dataset1 = df_amazon[["id", "title", "description", "manufacturer", "price"]]
    dataset2 = df_google[["id", "title", "description", "manufacturer", "price"]]

    print(f"Carregados {len(dataset1)} produtos da Amazon e {len(dataset2)} do Google.")

    weights = {"title": 0.97, "description": 0.01, "manufacturer": 0.01, "price": 0.01}
    G = build_similarity_graph(dataset1, dataset2, similarity_func="cosine", weights=weights)

    clusters, unmatched_V1, unmatched_V2 = unique_mapping_clustering(G, t=0.25)

    export_clusters(clusters, "umc_results.csv")

    gt_path = cleaned_dir / "cleaned_Amzon_GoogleProducts_perfectMapping.csv"
    gt_df = pd.read_csv(gt_path)
    ground_truth = set(zip(gt_df["idAmazon"].astype(str), gt_df["idGoogleBase"].astype(str)))

    metrics = evaluate_clusters(clusters, ground_truth)

    print(f"Total de clusters formados: {len(clusters)}")
    print(f"Produtos não pareados na Amazon: {len(unmatched_V1)}")
    print(f"Produtos não pareados no Google: {len(unmatched_V2)}")
    print(f"Paridade Amazon: {metrics['parity_dataset1']:.2%}")
    print(f"Paridade Google: {metrics['parity_dataset2']:.2%}")
    print(
        f"Precisão: {metrics['precision']:.2%} "
        f"Recall: {metrics['recall']:.2%} "
        f"F1: {metrics['f1']:.2%}"
    )

