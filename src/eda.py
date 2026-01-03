import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/churn.csv")
df.columns = df.columns.str.strip()

# 1️⃣ Churn distribution
sns.countplot(x="Customer Status", data=df)
plt.title("Customer Status Distribution")
plt.show()

# 2️⃣ Tenure vs Churn
sns.boxplot(x="Customer Status", y="Tenure in Months", data=df)
plt.title("Tenure vs Customer Status")
plt.show()

# 3️⃣ Monthly charge vs Churn
sns.boxplot(x="Customer Status", y="Monthly Charge", data=df)
plt.title("Monthly Charge vs Customer Status")
plt.show()
