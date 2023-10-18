"""
exercise demonstating clustering methods k means clustering and BDSCAN
and dimensional reduction (PCA)
"""

# Import modules
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier



titanic_train = pd.read_csv("../data/train.csv")
titanic_test = pd.read_csv("../data/test.csv")

# filtering train

# replace male and female with integer values
titanic_train = titanic_train.replace(["male", "female"], [0, 1])

# drop unwanted coloumns

titanic_train = titanic_train.drop(["Cabin", "Ticket", "Name", "Embarked", "PassengerID"], axis=1)

titanic_train["Age"] = titanic_train["Age"].fillna().mean()

train_x = titanic_train.drop("Survived", axis=1)
train_y = titanic_train["Survived"]


# filtering test


# replace male and female with integer values
titanic_test = titanic_test.replace(["male", "female"], [0, 1])

# drop unwanted coloumns

titanic_test = titanic_test.drop(["Cabin", "Ticket", "Name", "Embarked", "PassengerID"], axis=1)

titanic_test["Age"] = titanic_test["Age"].fillna().mean()
titanic_test["Fare"] = titanic_test["Fare"].fillna().mean()

# knn_clf=KNeighborsClassifier()
# knn_clf.fit(churnx_train, churny_train)
# ypred = knn_clf.predict(churnx_test) # These are the predicted output values
