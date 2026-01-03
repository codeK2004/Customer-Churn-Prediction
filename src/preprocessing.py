import pandas as pd

df = pd.read_csv("data/churn.csv")
df.columns = df.columns.str.strip()

# 🔹 Keep only customers who Stayed or Churned
df = df[df["Customer Status"].isin(["Stayed", "Churned"])]

# 🔹 Create binary churn label
df["Churn"] = df["Customer Status"].map({
    "Stayed": 0,
    "Churned": 1
})

# 🔹 Select useful features only
features = [
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Contract",
    "Payment Method",
    "Internet Service",
    "Gender",
    "Married",
    "Number of Dependents"
]

df = df[features + ["Churn"]]

# 🔹 Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# 🔹 Drop missing values
df.dropna(inplace=True)

# 🔹 Save clean dataset
df.to_csv("data/clean_churn.csv", index=False)

print("Preprocessing complete")
print(df.head())
