# Importing Required Libraries
# Pandas and NumPy are used for data manipulation, while Matplotlib and Seaborn are for data visualization.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Insurance Dataset
df = pd.read_csv('insurance.csv')

# Display the first few rows of the dataset to understand its structure
df.head()

# Check the shape of the dataset (number of rows and columns)
print("Dataset Shape:", df.shape)

# Display basic information about the dataset, including data types and non-null counts
df.info()

# Check for missing values in the dataset
missing_values = df.isnull().any()
print("Missing Values:", missing_values)

# Check for duplicate rows in the dataset and remove them if present
def check_duplicates(data):
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"Number of duplicate rows: {duplicates}")
        return data.drop_duplicates()
    else:
        print("No duplicate rows found.")
        return data

df = check_duplicates(df)

# Statistical Summary of the Dataset
print(df.describe())

# Correlation Analysis for Numeric Columns
# Correlation helps to identify relationships between numeric variables.
correlation = df.corr(numeric_only=True)
print("Correlation Matrix:")
print(correlation)

# --- Data Visualization ---
# Visualizing relationships and distributions for deeper insights.

# Scatter Plot: Age vs Charges (colored by smoker status and styled by sex)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='charges', data=df, style='sex', hue='smoker')
plt.title('Scatter Plot: Age vs Charges')
plt.savefig('scatterplot_age_vs_charges.png')
plt.show()

# Scatter Plot: Age vs BMI (colored by smoker status and styled by sex)
plt.figure(figsize=(10, 8))
sns.scatterplot(x='age', y='bmi', data=df, style='sex', hue='smoker')
plt.title('Scatter Plot: Age vs BMI')
plt.savefig('scatterplot_age_vs_bmi.png')
plt.show()

# Bar Plot: Sex vs Charges (grouped by smoker status)
plt.figure(figsize=(7, 6))
sns.barplot(x='sex', y='charges', hue='smoker', data=df)
plt.title('Bar Plot: Sex vs Charges')
plt.savefig('barplot_sex_vs_charges.png')
plt.show()

# Bar Plot: Region vs Charges (grouped by smoker status)
plt.figure(figsize=(7, 6))
sns.barplot(x='region', y='charges', hue='smoker', data=df)
plt.title('Bar Plot: Region vs Charges')
plt.savefig('barplot_region_vs_charges.png')
plt.show()

# KDE Plot: Distribution of Charges
plt.figure(figsize=(7, 8))
sns.kdeplot(df['charges'], shade=True)
plt.title('KDE Plot: Charges Distribution')
plt.grid()
plt.savefig('kdeplot_charges_distribution.png')
plt.show()

# KDE Plot: Distribution of BMI
plt.figure(figsize=(7, 8))
sns.kdeplot(df['bmi'], shade=True)
plt.title('KDE Plot: BMI Distribution')
plt.grid()
plt.savefig('kdeplot_bmi_distribution.png')
plt.show()

# Heatmap: Correlation Between Categorical Variables (Region and Children)
df['region_encoded'] = df['region'].astype('category').cat.codes
heatmap_data = df[['children', 'region_encoded']].corr()
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
plt.title('Heatmap: Children vs Region (Encoded)')
plt.savefig('heatmap_children_vs_region.png')
plt.show()

# Heatmap: Correlation Between Smoker and Sex
df['smoker_encoded'] = df['smoker'].astype('category').cat.codes
df['sex_encoded'] = df['sex'].astype('category').cat.codes
heatmap_data = df[['sex_encoded', 'smoker_encoded']].corr()
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
plt.title('Heatmap: Sex vs Smoker (Encoded)')
plt.savefig('heatmap_sex_vs_smoker.png')
plt.show()

# Final Cleaned Dataset
print("Final Dataset:")
print(df.head())
