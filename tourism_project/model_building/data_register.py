import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(RAW_PATH)

expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch",
]

missing = [column for column in expected_columns if column not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns in CSV: {df.shape[1]}")
print("Columns:", list(df.columns))
print("\nTarget distribution:")
print(df["ProdTaken"].value_counts(dropna=False))
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))
