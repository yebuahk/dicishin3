import statistics as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
df_swa = pd.read_csv('Social_Network_Ads.csv')
df.head()

df_swa.head()

X = df_swa.iloc[:, [2, 3]]
X.head()

X.isnull().any().any()

y = df_swa.iloc[:, 4]
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

X_train.shape

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
cm = confusion_matrix(y_pred, y_test)
cm

pd.crosstab(y_pred, y_test)

total = sum(sum(cm))

accuracy = (cm[0, 0]+cm[1, 1])/total
print('Accuracy :', accuracy)
classifier.score(X_test, y_test)    # calculate accuracy automatically instead of manually

sensitivity = cm[0, 0]/(cm[0, 0]+cm[0, 1])
print('Sensitivity :', sensitivity)

specificity = cm[1, 1]/(cm[1, 0]+cm[1, 1])
print('specificity :', specificity)

precision = cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('precision :', precision)

print(classification_report(y_test, y_pred))

dtree = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=10, min_samples_leaf=5, max_depth=5)
dtree.fit(X_train, y_train)

predict3 = dtree.predict(X_test)
score2 = dtree.score(X_test, y_test)
score2

from sklearn import tree
from IPython.display import Image

tree.export_graphviz(classifier, out_file='C:/Users/kevin/PycharmProjects/dicishin3/wp.png')
Image(filename='C:/Users/kevin/PycharmProjects/dicishin3/wp.png')