import polars as pl

def sample_balanced(df, label_col, n_per_class, seed=1) -> pl.DataFrame:
    """
    Returns a balanced subset with n_per_class samples per class.
    If a class has fewer than n_per_class samples, all samples are taken.
    """
    return (
        df.collect().group_by(label_col).map_groups(lambda group: group.sample(
            n=min(n_per_class, len(group)),
            seed=seed
        ))
        .sample(fraction=1.0, shuffle=True, seed=seed)  # shuffle final result
    )