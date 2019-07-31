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
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

X_train.shape

y_pred = classifier.predict(X_test)
