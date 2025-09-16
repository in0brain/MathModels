import pandas as pd

def clean_table(df: pd.DataFrame, dropna_cols=None, dedup=True, rename_map=None) -> pd.DataFrame:
    if rename_map: df = df.rename(columns=rename_map)
    if dropna_cols: df = df.dropna(subset=dropna_cols)
    if dedup: df = df.drop_duplicates()
    return df
