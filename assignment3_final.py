## import

import numpy as np
import pandas as pd

# Get the data into pandas
df_t = pd.read_csv("data_scientist_duplicate_detection.csv")
df = df_t.copy()

# Explore the data
print(df.nunique(),"\n", df.isna().sum(),"\n",df.dtypes, "\n", df.id.duplicated(keep=False).sum())

# Group by, Transform and count
df["count"] = df.groupby(["name","country","city"])["id"].transform('count')
print("Number of duplicate rows", len(df[df["count"]>1]))

# Make name country city into one column to clearly see the duplicates
df["Name-Country-City"] = df["name"] + "-" + df["country"] + "-" +  df["city"]

# Filter based on count and add column of fields they are duplicated on:
df.loc[df["count"]>1,["dup_rows"]] = "name-Country-city"

print("Number of duplicates final check", df[df.dup_rows.notna()].shape)

## push to CSV

df[df.dup_rows.notna()].to_csv("duplicates.csv", index=False)
