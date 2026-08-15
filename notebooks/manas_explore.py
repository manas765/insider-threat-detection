import pandas as pd

# Change this path to wherever your logon.csv actually is
path = r"C:\Users\manas\Downloads\answers\answers\insiders.csv"

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())