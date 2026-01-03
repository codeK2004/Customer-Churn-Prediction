import pandas as pd

# Load dataset
df = pd.read_csv("data/churn.csv")

print(df.head())
print(df.info())
