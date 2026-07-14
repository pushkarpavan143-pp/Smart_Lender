import pandas as pd

# Path to your dataset
file_path = "Dataset/loan_prediction.csv"

print("Attempting to read your CSV dataset...")

try:
    # Read the dataset with clean fallback encoding
    df = pd.read_csv(file_path, encoding='unicode_escape')
    
    print("SUCCESS: Python successfully read your CSV file.")
    print("Total Rows: " + str(df.shape[0]) + " | Total Columns: " + str(df.shape[1]))
    
    print("Columns available in your dataset:")
    print(df.columns.tolist())
    
    print("Here is a sneak peek at the first 3 rows:")
    print(df.head(3))

except FileNotFoundError:
    print("ERROR: Could not find the file at " + file_path)
    print("Check your Dataset folder sidebar to make sure the file name matches perfectly.")
except Exception as e:
    print("ERROR: Something went wrong.")
    print("Details: " + str(e))