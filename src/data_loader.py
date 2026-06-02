import polars as pl
from config import RAW_DIR

def load_lazy(path):
    """ Load all the dataset from a parquet file """
    df = pl.scan_parquet(path)

    return df


def load_columns(path, columns):
    """ Load selected columns from a parquet file
    Parameters:
        path: str -> path to parquet file
        columns: list[str] -> columns to load
    """
    df = pl.scan_parquet(path).select(columns)

    return df



