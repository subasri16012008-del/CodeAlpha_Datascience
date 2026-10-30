
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON

# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
df = pd.read_csv("Unemployment in India.csv")

# Step 3: Clean column names
df.columns = df.columns.str.strip()

# Step 4: Display dataset
print("FIRST FIVE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

# Step 5: Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 6: Remove duplicate records
df.drop_duplicates(inplace=True)

# Step 7: Handle missing values
df = df.dropna(subset=[
    "Date",
    "Estimated Unemployment Rate (%)"
])

# Step 8: Convert date column
df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=True, errors="coerce"
)

df = df.dropna(subset=["Date"])

# Step 9: Convert unemployment rate to numeric
df["Estimated Unemployment Rate (%)"] = pd.to_numeric(
    df["Estimated Unemployment Rate (%)"],
    errors="coerce"
)

df.dropna(
    subset=["Estimated Unemployment Rate (%)"],
    inplace=True
)

# Step 10: Extract year and month
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()

# Step 11: Display statistical information
print("\nSTATISTICAL DESCRIPTION")
print(df.describe())

# Step 12: Calculate overall unemployment rate
average_rate = df["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Rate:",
      round(average_rate, 2), "%")

# Step 13: Year-wise unemployment analysis
yearly_data = df.groupby("Year")[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nYEAR-WISE UNEMPLOYMENT RATE")
print(yearly_data)

# Step 14: Monthly unemployment analysis
monthly_data = df.groupby("Month_Name")[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nMONTH-WISE UNEMPLOYMENT RATE")
print(monthly_data)

# Step 15: COVID-19 impact analysis
before_covid = df[
    df["Date"] < "2020-03-01"
]

during_covid = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-12-31")
]

after_covid = df[
    df["Date"] > "2020-12-31"
]

print("\nCOVID-19 IMPACT ANALYSIS")

print("Before COVID:",
      round(before_covid[
          "Estimated Unemployment Rate (%)"
      ].mean(), 2), "%")

print("During COVID:",
      round(during_covid[
          "Estimated Unemployment Rate (%)"
      ].mean(), 2), "%")

if not after_covid.empty:
    print("After COVID:",
          round(after_covid[
              "Estimated Unemployment Rate (%)"
          ].mean(), 2), "%")

# Step 16: Region-wise unemployment analysis
if "Region" in df.columns:
    region_data = df.groupby("Region")[
        "Estimated Unemployment Rate (%)"
    ].mean().sort_values(ascending=False)

    print("\nREGION-WISE UNEMPLOYMENT RATE")
    print(region_data)

    plt.figure(figsize=(12, 6))
    region_data.plot(kind="bar")
    plt.title("Region-wise Unemployment Rate")
    plt.xlabel("Region")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Step 17: Visualize unemployment trend
plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 18: Year-wise bar chart
plt.figure(figsize=(10, 5))

yearly_data.plot(kind="bar")

plt.title("Year-wise Unemployment Rate")
plt.xlabel("Year")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# Step 19: Monthly trend visualization
plt.figure(figsize=(10, 5))

sns.barplot(
    x="Month",
    y="Estimated Unemployment Rate (%)",
    data=df,
    errorbar=None
)

plt.title("Monthly Unemployment Trends")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# Step 20: COVID-19 comparison
covid_labels = ["Before COVID", "During COVID"]

covid_values = [
    before_covid["Estimated Unemployment Rate (%)"].mean(),
    during_covid["Estimated Unemployment Rate (%)"].mean()
]

plt.figure(figsize=(8, 5))

plt.bar(covid_labels, covid_values)

plt.title("Impact of COVID-19 on Unemployment")
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# Step 21: Correlation analysis
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(9, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()

# Step 22: Final observations
print("\nFINAL ANALYSIS")

print("Average Unemployment Rate:",
      round(average_rate, 2), "%")

if not before_covid.empty and not during_covid.empty:
    difference = (
        during_covid["Estimated Unemployment Rate (%)"].mean()
        - before_covid["Estimated Unemployment Rate (%)"].mean()
    )

    print("COVID-19 Unemployment Rate Change:",
          round(difference, 2), "percentage points")

    if difference > 0:
        print("Unemployment increased during COVID-19.")
    else:
        print("Unemployment did not increase during COVID-19.")

print("\nUNEMPLOYMENT ANALYSIS COMPLETED SUCCESSFULLY!")
