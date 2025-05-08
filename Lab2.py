import pandas as pd
from sklearn.metrics import (precision_score, recall_score)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('processed_titanic.csv')
print(df.head())
print(df.dtypes)
print('\n')
df.drop(columns=["Name"], inplace=True)

Xr = df.drop(columns=['Age', 'Sex', 'Ticket', 'Cabin'])
yr = df['Age']

Xcl = df.drop(columns=['Survived', 'PassengerId', 'Sex', 'Ticket', 'Cabin'])
ycl = df['Survived']

X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(Xr, yr, test_size=0.4, random_state=42)
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(Xcl, ycl, test_size=0.4, random_state=42)

logreg_model = LogisticRegression(max_iter=5000)
logreg_model.fit(X_log_train, y_log_train)
y_log_result = logreg_model.predict(X_log_test)
cm_log = confusion_matrix(y_log_test, y_log_result)

linear_model = LinearRegression()
linear_model_class = LinearRegression()
linear_model.fit(X_lin_train, y_lin_train)
linear_model_class.fit(X_log_train, y_log_train)
y_lin_result = linear_model.predict(X_lin_test)
y_lin_result_binary = [1 if x >= 0.5 else 0 for x in y_lin_result]
y_lin_test_binary = [1 if x >= 0.5 else 0 for x in y_lin_test]
cm_lin = confusion_matrix(y_lin_test_binary, y_lin_result_binary)


print("Logistic regression")
print("Precision:", precision_score(y_log_test, y_log_result))
print("Recall:", recall_score(y_log_test, y_log_result))
print('\n')

print("Linear regression")
print("Precision:", precision_score(y_lin_test_binary, y_lin_result_binary))
print("Recall:", recall_score(y_lin_test_binary, y_lin_result_binary))
print('\n')

plot.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Reds')
plot.title('Confusion matrix for logistic regression')
plot.ylabel('True label')
plot.xlabel('Predicted label')
plot.show()

plot.figure(figsize=(4, 4))
sns.heatmap(cm_lin, annot=True, fmt='d', cmap='Greens')
plot.title('Confusion matrix for linear regression')
plot.ylabel('True label')
plot.xlabel('Predicted label')
plot.show()