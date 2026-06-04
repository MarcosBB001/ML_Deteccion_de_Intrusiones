import polars as pl
from src.data_loader import load_lazy, load_columns
from src.config import CIC_TRAIN_PATH, CIC_TEST_PATH

train_df = load_lazy(CIC_TRAIN_PATH)
test_df = load_lazy(CIC_TEST_PATH)

print(train_df)
print(train_df.head().collect())

# Label attack class disitribution
train_dist = (train_df.group_by("attack_class").len().sort("len", descending=True).collect())
test_dist = (test_df.group_by("attack_class").len().sort("len", descending=True).collect())

print(train_dist)
print(test_dist)

# Benign (0) and Malign (1) label distribution
train_binary = train_df.group_by("label").len().collect()
test_binary = test_df.group_by("label").len().collect()

print(train_binary)
print(test_binary)

# Some numeric columns info
cols = ["Rate", "IAT", "Tot size", "syn_flag_number", "ack_flag_number"]

df_num = load_columns(CIC_TRAIN_PATH, cols)

print(df_num.collect().describe())