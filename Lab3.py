import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plot

df = pd.read_csv(r"processed_titanic.csv")

X = df.drop(columns=['Survived','Name','Sex', 'Ticket','Cabin'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

cm = confusion_matrix(y_test, y_pred)
plot.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plot.title('Confusion matrix for logistic regression')
plot.ylabel('True label')
plot.xlabel('Predicted label')
plot.show()

plot.figure(figsize=(20, 10))
plot_tree(clf, filled = True,  feature_names = X.columns, class_names = ['Not Survived', 'Survived'], rounded=True, proportion=True, precision=2)
plot.title('Solution tree')
plot.show()