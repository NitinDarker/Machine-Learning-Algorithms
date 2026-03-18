import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('placement.csv')
# print(df)

# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

df['CGPA'] = df['CGPA'].astype(str).str.strip()
df['Package'] = df['Package'].astype(str).str.strip()

df['CGPA'] = pd.to_numeric(df['CGPA'], errors='coerce')
df['Package'] = pd.to_numeric(df['Package'], errors='coerce')



df = df.dropna()
print(df.isnull().sum())
print(df.shape)


df = df[(df['CGPA'] >= 0) & (df['CGPA'] <= 10)]
df = df[df['Package'] >= 0]


df = df.drop_duplicates()

print(df.shape)


# plt.figure(figsize=(10,5))
# plt.hist(df['CGPA'], bins=20, color='skyblue')
# plt.title("CGPA Distribution")
# plt.xlabel("CGPA")
# plt.ylabel("Frequency")
# plt.show()


# plt.figure(figsize=(10,5))
# plt.hist(df['Package'], bins=20, color='orange')
# plt.title("Package Distribution")
# plt.xlabel("Package")
# plt.ylabel("Frequency")
# plt.show()

# plt.figure(figsize=(8,6))
# sns.scatterplot(x='CGPA', y='Package', data=df)
# plt.title("CGPA vs Package")
# plt.show()


# plt.figure(figsize=(8,4))
# sns.boxplot(x=df['Package'])
# plt.show()


# plt.figure(figsize=(8,4))
# sns.boxplot(x=df['CGPA'])
# plt.show()



Q1 = df['Package'].quantile(0.25)
Q3 = df['Package'].quantile(0.75)
IQR = Q3 - Q1


lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR


df = df[(df['Package'] >= lower) & (df['Package'] <= upper)]

# sns.boxplot(x=df['Package'])
# plt.show()

# sns.scatterplot(x='CGPA', y='Package', data=df)
# plt.show()

# print(df.corr())
df.to_csv("cleaned_data.csv", index=False)
