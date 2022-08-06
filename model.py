
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data.csv")

X = df[["Height", "Weight", "Eye"]]
X = X.replace(["Brown", "Blue"], [1, 0])

y = df["Species"]

clf = LogisticRegression() 
clf.fit(X, y)

pickle.dump(clf, open("clf.pkl", 'wb'))
