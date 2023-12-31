"""
exercise demonstating clustering methods k means clustering and BDSCAN
and dimensional reduction (PCA)
"""

# Import modules
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from perform_pca import perform_pca
import os
import numpy as np

print(os.getcwd())

pca_option = True

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

scores_test = {}  # save scores in empty dictionary
ypred = {}  # save predictions in emty dictionary
scores_train = {}

# Go through 50 k-means without PCA
for i in range(1, 51):
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(train_x, train_y)
    ypred[i] = knn_clf.predict(titanic_test)
    scores_test[i] = knn_clf.score(titanic_test, y_test)
    scores_train[i] = knn_clf.score(train_x, train_y)
    print("k = {}, Test Accuracy: {:.2f}, Training Accuracy: {:.2f}".format(i, scores_test[i], scores_train[i]))

    # Plot the scores in a lineplot
fig, ax = plt.subplots()
ax.plot(range(1, 51), scores_test.values(), label="Genauigkeit des Testdatensatzes")
ax.plot(range(1, 51), scores_train.values(), label="Genauigkeit des Trainingsdatensatzes")
ax.set_title('Genauigkeit für verschiedene Werte von k')
ax.set_xlabel('Anzahl der Nachbarn (k)')
ax.set_ylabel('Genauigkeit')
plt.legend()

ax.grid(True)

# Save the plot without PCA
plt.savefig("../output/kmeansscoreplot.png")
plt.show()

# Function to extract the score for comparison
def get_score(item):
    return item[1]

# Find the PCA component with the highest test score
best_pca_component, highest_test_score = max(scores_test.items(), key=get_score)

# Print the best PCA component and its corresponding test score
print("Best PCA Komponenten: {}, Highest Test Score: {:.2f}".format(best_pca_component, highest_test_score))


# Define a range of PCA components to test
pca_components_range = range(1, 7)  # only 6 smaples = only 6 max components

# Store the results in a dictionary
results = {}

# Define the number of rows and columns for the subplot matrix
n_rows = 3
n_cols = 2

# Create a figure and axis objects for the subplot matrix
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

# Initialize a counter for the subplot
subplot_counter = 0

if pca_option:
    for j in pca_components_range:
        if j <= min(train_x.shape[0], train_x.shape[1]):
            # Create a copy of the original train_x and titanic_test
            train_x_pca = train_x.copy()
            titanic_test_pca = titanic_test.copy()

            # Apply PCA with 'j' components to both the training and test data
            train_x_pca = perform_pca(train_x_pca, j)
            titanic_test_pca = perform_pca(titanic_test_pca, j)

            # Make sure train_y remains unchanged
            # Go through 50 k-NN iterations for the PCA-transformed data
            for i in range(1, 51):
                knn_clf = KNeighborsClassifier(n_neighbors=i)
                knn_clf.fit(train_x_pca, train_y)
                accuracy = knn_clf.score(titanic_test_pca, y_test)  # Use transformed test data
                results[(j, i)] = accuracy

            # Plot the PCA components
            row = subplot_counter // n_cols
            col = subplot_counter % n_cols
            axes[row, col].plot(range(1, 51), [results[(j, i)] for i in range(1, 51)], marker='o')
            axes[row, col].set_title(f"PCA Komponente: {j}")
            axes[row, col].set_xlabel("Anzahl der Nachbarn (k)")
            axes[row, col].set_ylabel("Genauigkeit")

            # Increment the subplot counter
            subplot_counter += 1

# Adjust layout to prevent overlap of titles and labels
plt.tight_layout()

plt.savefig("../output/pcakmeans_plots.png")
# Show the plots
plt.show()

# Print and analyze the results
for (j, i), accuracy in results.items():
    print("PCA Komponenten: {}, k-NN Nachbarn: {}, Genauigkeit: {:.2f}".format(j, i, accuracy))


# Store the results in a dictionary
results = {}

# Define the number of rows and columns for the subplot matrix
n_rows = 3
n_cols = 2

# Create a figure and axis objects for the subplot matrix
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

# Initialize a counter for the subplot
subplot_counter = 0
error_rate = []

if pca_option:
    for j in pca_components_range:
        if j <= min(train_x.shape[0], train_x.shape[1]):
            # Create a copy of the original train_x and titanic_test
            train_x_pca = train_x.copy()
            titanic_test_pca = titanic_test.copy()

            # Apply PCA with 'j' components to both the training and test data
            train_x_pca = perform_pca(train_x_pca, j)
            titanic_test_pca = perform_pca(titanic_test_pca, j)

            # Make sure train_y remains unchanged
            # Go through 50 k-NN iterations for the PCA-transformed data
            for i in range(1, 51):
                knn_clf = KNeighborsClassifier(n_neighbors=i)
                knn_clf.fit(train_x_pca, train_y)
                # Convert y_test DataFrame to a NumPy array
                y_test_array = y_test.values.flatten()  # Flattening to ensure it's a 1D array

                y_pred = knn_clf.predict(titanic_test_pca)

                # Now, you can calculate the error rate without dimension mismatch
                error_rate.append(np.mean(y_pred != y_test_array))  # Calculate error rate
                results[(j, i)] = error_rate[-1]

            # Plot the PCA components
            row = subplot_counter // n_cols
            col = subplot_counter % n_cols
            axes[row, col].plot(range(1, 51), [results[(j, i)] for i in range(1, 51)], marker='o')
            axes[row, col].set_title(f"PCA Komponente: {j} - Error Rate")
            axes[row, col].set_xlabel("Anzahl der Nachbarn (k)")
            axes[row, col].set_ylabel("Fehler Rate")

            # Increment the subplot counter
            subplot_counter += 1

# Adjust layout to prevent overlap of titles and labels
plt.tight_layout()

plt.savefig("../output/pcakmeans_error_plots.png")
# Show the plots
plt.show()

# Print and analyze the results
for (j, i), error_rate in results.items():
    print("PCA Komponenten: {}, k-NN Nachbarn: {}, Fehler Rate: {:.2f}".format(j, i, error_rate))

# Function to extract error rate from a tuple
def get_error_rate(result):
    return result[1] # get second element of the tupel = error-rate

# Find the combination with the lowest error rate
best_components, min_error_rate = min(results.items(), key=get_error_rate)

# Print the combination with the lowest error rate
print("Lowest Fehler Rate: {:.2f}".format(min_error_rate))
print("Best PCA Komponenten: {}, Best k-NN Nachbarn: {}".format(best_components[0], best_components[1]))

