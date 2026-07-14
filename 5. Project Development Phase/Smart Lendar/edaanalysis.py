import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. IMPORT AND READ DATASET
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
csv_path = os.path.join(BASE_DIR, 'dataset', 'loan_approval.csv')

print(f" Reading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

print("\n--- Dataset Quick Look ---")
print(df.head())
print("\n--- Summary Statistics ---")
print(df.describe())

# Create a folder to save plots
os.makedirs('eda_plots', exist_ok=True)

# Global plotting style
sns.set_theme(style="whitegrid")

# =====================================================================
# 2. UNIVARIATE ANALYSIS
# =====================================================================
print("\nGenerating Univariate Analysis...")

# Distribution of Credit Scores
plt.figure(figsize=(8, 4))
sns.histplot(df['credit_score'], kde=True, color='skyblue')
plt.title('Univariate Analysis: Distribution of Credit Scores')
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_plots/univariate_credit_score.png')
plt.close()

# Loan Approval Status counts
if 'loan_approved' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='loan_approved', hue='loan_approved', data=df, palette='Set2', legend=False)
    plt.title('Univariate Analysis: Loan Approval Balance (0 = Denied, 1 = Approved)')
    plt.xlabel('Loan Approved Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('eda_plots/univariate_loan_status.png')
    plt.close()

# =====================================================================
# 3. BIVARIATE ANALYSIS
# =====================================================================
print("Generating Bivariate Analysis...")

# Income vs Loan Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(x='income', y='loan_amount', data=df, alpha=0.7, color='purple')
plt.title('Bivariate Analysis: Income vs Requested Loan Amount')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.tight_layout()
plt.savefig('eda_plots/bivariate_income_vs_loan.png')
plt.close()

# Credit Score vs Loan Approval
if 'loan_approved' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='loan_approved', y='credit_score', hue='loan_approved', data=df, palette='Pastel1', legend=False)
    plt.title('Bivariate Analysis: Credit Score Distribution by Approval Status')
    plt.xlabel('Loan Approved (0 = No, 1 = Yes)')
    plt.ylabel('Credit Score')
    plt.tight_layout()
    plt.savefig('eda_plots/bivariate_credit_vs_approval.png')
    plt.close()

# =====================================================================
# 4. MULTIVARIATE ANALYSIS
# =====================================================================
print("Generating Multivariate Analysis...")

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Multivariate Analysis: Feature Correlation Heatmap Matrix')
plt.tight_layout()
plt.savefig('eda_plots/multivariate_correlation_matrix.png')
plt.close()

# Income vs Loan Amount segmented by Loan Approval
if 'loan_approved' in df.columns:
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x='income', y='loan_amount', hue='loan_approved', data=df, palette='coolwarm', alpha=0.8)
    plt.title('Multivariate Analysis: Income vs Loan Amount colored by Approval Status')
    plt.xlabel('Annual Income')
    plt.ylabel('Loan Amount')
    plt.legend(title='Loan Approved')
    plt.tight_layout()
    plt.savefig('eda_plots/multivariate_scatter_with_hue.png')
    plt.close()

print("\n Done! All figures saved in 'eda_plots/' folder.")
