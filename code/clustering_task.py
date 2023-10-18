"""
exercise demonstating clustering methods k means clustering and BDSCAN
and dimensional reduction (PCA)
"""

# Import modules
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

titanic_train = pd.read_csv("../data/train.csv")
titanic_test = pd.read_csv("../data/test.csv")
y_test = pd.read_csv("../data/gender_submission.csv")
y_test = y_test.drop("PassengerId", axis=1)
# filtering train

# replace male and female with integer values
titanic_train = titanic_train.replace(["male", "female"], [0, 1])

# drop unwanted coloumns

titanic_train = titanic_train.drop(["Cabin", "Ticket", "Name", "Embarked", "PassengerId"], axis=1)

titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].mean())

train_x = titanic_train.drop("Survived", axis=1)
train_y = titanic_train["Survived"]

# filtering test


# replace male and female with integer values
titanic_test = titanic_test.replace(["male", "female"], [0, 1])

# drop unwanted coloumns

titanic_test = titanic_test.drop(["Cabin", "Ticket", "Name", "Embarked", "PassengerId"], axis=1)

titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].mean())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].mean())  # has one NA that needs to be filled.

scores = {}  # save scores in empty dictionary
ypred = {}  # save predictions in emty dictionary

for i in range(1, 51):
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(train_x, train_y)
    ypred[i] = knn_clf.predict(titanic_test)  # These are the predicted output values
    scores[i] = knn_clf.score(titanic_test, y_test)
    print("Genauigkeit auf dem Testdatensatz:{:.2f}".format(knn_clf.score(titanic_test, y_test)))

# plot the scores in a lineplot

# Create a new figure
fig, ax = plt.subplots()

# Plot the data
ax.plot(range(1, 51), scores.values(), marker='o')

# Set title and labels
ax.set_title('Genauigkeit für verschiedene Werte von k')
ax.set_xlabel('Anzahl der Nachbarn (k)')
ax.set_ylabel('Genauigkeit')

# Enable the grid
ax.grid(True)

# save plot-output
plt.savefig("../output/kmeansscoreplot.png")
# Show the plot
plt.show()
