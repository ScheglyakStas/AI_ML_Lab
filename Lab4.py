import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv(r"processed_titanic.csv")

X = df.drop(columns=['Survived','Name','Sex', 'Ticket','Cabin','Age']) #признаки
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

gb = GradientBoostingClassifier(random_state = 42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print("Random forest parameters")
print("F1-Score:", f1_score(y_test, rf_pred, average = 'weighted'))
print("Recall:", recall_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print('\n')

print("Gradient boosting parameters")
print("F1-Score:", f1_score(y_test, gb_pred, average = 'weighted'))
print("Recall:", recall_score(y_test, gb_pred))
print("Precision:", precision_score(y_test, gb_pred))


