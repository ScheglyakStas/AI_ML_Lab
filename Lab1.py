import pandas as pd
df = pd.read_csv("titanic.csv")  # путь к файлу
df.head()  # просмотр первых строк
print(pd.isna(df))

df["Cabin"].fillna(df["Cabin"].mode()[0], inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()  # или StandardScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["Fare"] = scaler.fit_transform(df[["Fare"]])
df["Pclass"] = scaler.fit_transform(df[["Pclass"]])

df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
df.to_csv("processed_titanic.csv", index=False)