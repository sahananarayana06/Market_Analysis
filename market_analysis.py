import pandas as pd

# Load the dataset
df = pd.read_csv("SampleSuperstore.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Check data types
print("\nData Types:")
print(df.dtypes)

# Display cleaned dataset information
print("\nCleaned Dataset Shape:", df.shape)

# Total Sales
print("\nTotal Sales:", df["Sales"].sum())

# Total Profit
print("Total Profit:", df["Profit"].sum())

# Average Sales
print("Average Sales:", df["Sales"].mean())

# Top 10 Products by Sales
top_products = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)

print("\nTop Products by Sales:")
print(top_products.head(10))

import matplotlib.pyplot as plt

# Top 10 Sub-Categories by Sales
top_products = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))
top_products.head(10).plot(kind="bar")
plt.title("Top 10 Sub-Categories by Sales")
plt.xlabel("Sub-Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sales Distribution by Category(Pie chart)
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(7,7))
plt.pie(category_sales,
        labels=category_sales.index,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Sales Distribution by Category")
plt.show()

# Profit by Region
region_profit = df.groupby("Region")["Profit"].sum()

plt.figure(figsize=(8,5))
region_profit.plot(kind="bar")

plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

#Relationship Between Quantity and Sales
plt.figure(figsize=(8,5))

plt.scatter(df["Quantity"], df["Sales"])

plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")

plt.show()

# Sales Prediction using Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Features and Target
X = df[['Quantity', 'Discount']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\nModel Performance")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Example prediction
sample = [[5, 0.10]]
predicted_sales = model.predict(sample)

print("\nPredicted Sales for Quantity=5 and Discount=10%:")
print(predicted_sales[0])