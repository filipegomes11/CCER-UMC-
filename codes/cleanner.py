import unicodedata
import pandas as pd
import glob
import os
import re

def clean_csv(file_path, output_dir):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')

    for col in df.columns:
        if "id" in col.lower():
            continue
        df[col] = df[col].map(
            lambda x: re.sub(r"[^a-zA-Z0-9\s.,;:!?@#%&\-_]", "", str(x)) if isinstance(x, str) else x
        )
        df[col] = df[col].map(remove_accents)

    df = df.fillna('')
    base = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"cleaned_{base}")
    df.to_csv(output_path, index=False)
    print(f"Cleaned file saved to {output_path}")

def remove_accents(text):
    if not isinstance(text, str):
        return text
    # Normaliza e remove marcas de acento
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def clean_all_csvs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    for file_path in csv_files:
        clean_csv(file_path, output_dir)
