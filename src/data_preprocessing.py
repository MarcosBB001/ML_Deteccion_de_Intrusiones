import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def encode_protocol_type(df, col="Protocol Type"):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    encoded = enc.fit_transform(df[col].to_numpy().reshape(-1, 1))
    ohe_cols = [f"Protocol_Type_{int(c)}" for c in enc.categories_[0]]
    
    df = df.drop(col).hstack(pl.DataFrame(encoded, schema=ohe_cols))
    
    return df


def scale_features(x_train, x_test, feature_cols, columns_to_scale, scaler_type="standard"):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    
    indices_to_scale = [feature_cols.index(col) for col in columns_to_scale]
    
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    
    x_train_scaled[:, indices_to_scale] = scaler.fit_transform(x_train[:, indices_to_scale])
    x_test_scaled[:, indices_to_scale] = scaler.transform(x_test[:, indices_to_scale])
    
    return x_train_scaled, x_test_scaled


