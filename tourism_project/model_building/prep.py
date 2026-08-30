import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(RAW_PATH)
df = df.drop_duplicates().reset_index(drop=True)

# Remove identifier/index columns that should not be used for prediction.
df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

target_col = "ProdTaken"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found.")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print("Training feature shape:", Xtrain.shape)
print("Testing feature shape :", Xtest.shape)
print("\nTraining target distribution:")
print(ytrain.value_counts())
