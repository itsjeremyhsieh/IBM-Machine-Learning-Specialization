# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import csv
# %%
df = pd.read_csv('Sleep_Efficiency.csv')
df.info()
#check null values
print(df.isnull().sum())
#replace null with median
df["Awakenings"] = df["Awakenings"].fillna(df["Awakenings"].median())
df["Caffeine consumption"] = df["Caffeine consumption"].fillna(df["Caffeine consumption"].median())
df["Alcohol consumption"] = df["Alcohol consumption"].fillna(df["Alcohol consumption"].median())
df["Exercise frequency"] = df["Exercise frequency"].fillna(df["Exercise frequency"].median())
#check for duplicate
print(df.duplicated().sum())

#data type convertion
df["Bedtime"] = pd.to_datetime(df["Bedtime"])
df["Wakeup time"] = pd.to_datetime(df["Wakeup time"])
df["Awakenings"] = df["Awakenings"].astype("int")
df["Exercise frequency"] = df["Exercise frequency"].astype("int")
print(df.info())
print(df.describe())
# %%
cols = ["Age", "Sleep duration", "Sleep efficiency", "REM sleep percentage", "Deep sleep percentage", "Light sleep percentage", "Awakenings", "Caffeine consumption", "Alcohol consumption", "Exercise frequency"]
fig,axes = plt.subplots(2,5, figsize=(25,15))

index = 0
for i in range(2):
  for j in range(5):
    sns.boxplot(y=cols[index], data=df, ax=axes[i,j])
    index += 1
# %%
out_cols = ["Light sleep percentage", "Deep sleep percentage", "Caffeine consumption", "Sleep duration"]
for col in out_cols:
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3-q1
  min = q1 - 1.5 * iqr
  max = q3 + 1.5 * iqr
  for x in range(0, len(df[col])):
    value = df[col][x] 
    if value  > max:
        value  = max
    if value  < min:
        value = min
    df[col][x] = value 

# %%
fig,axes = plt.subplots(1,4, figsize=(25,15))
index = 0
for i in range(4):
    sns.boxplot(y=out_cols[index], data=df, ax=axes[i])
    index += 1
# %%
df.to_csv('cleaned_data.csv', index=False)
# %%
sns.relplot(
    data=df, kind="line",
    x="Age", y="Sleep efficiency", hue="Gender")
plt.show()
# %%
plt.plot(df["Sleep efficiency"])

plt.xlabel("Yes or No", color="green",fontsize=10)
plt.ylabel("Count", color="green",fontsize=10)
plt.title("number of smokers and non-smokers", color="green",fontsize=10)
plt.show()
# %%
