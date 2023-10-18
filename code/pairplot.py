# Bibliotheken laden
import seaborn as sns
import matplotlib.pyplot as plt

# Datensatz aus der Bibliothek seaborn laden
df = sns.load_dataset("iris")

# Eine erste Übersicht über den Datensatz
print(df.head())
print(df.describe())

# Den Pairplot erstellen und anzeigen
sns.pairplot(df, hue="species")
plt.show()

