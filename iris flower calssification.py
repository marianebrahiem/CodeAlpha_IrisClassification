import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load datset

iris = load_iris()
x = iris.data # feature: sepal lenght, sepal width, petal lenght, petal width
y = iris.target # labels: 0=setose, 1=  versicolotr, 2=virginice

#convert to datFrame
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]
print("First 5 rows of the dataset:\n",df.head())

#save datset to csv
df.to_csv("iris_datset.csv", index=False)
print("Dataset saved as iris_dataset.csv")

#Data Visualization

sns.pairplot(df, hue="species", markers=["o","s","D"])
plt.suptitle("Iris Dataset Feature Relationship", y=1.02)
plt.show()

#Boxplot for sepal lenght distribution

plt.figure(figsize=(8,6))
sns.boxplot(x="species", y="sepal lenght (cm)", data=df)
plt.title("Sepal Lenght Distribution by Species")


#split data into training and testing sets


x_train,x_test,y_train,y_test =train_test_split(
    x, y, test_size=0.2, random_state=42
)

#feature scaling


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train KNN Claasifier


model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

#Make predictions

y_pred = model.predict(x_test)

#Evaluate the model

accuracy= accuracy_score(y_test, y_pred)
print("\nMpdel Accuracy:", accuracy)
print("\nClassification Report:\n",classification_report(y_test,y_pred))

#Confirm Matric Visualization

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()