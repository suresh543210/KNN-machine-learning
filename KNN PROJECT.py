 
import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv("F:\MEACHINE LEARNING\Data Set\Iris_new.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

# Define the GUI
root = tk.Tk()
root.title("KNN Classifier")

# Define the input fields
label1 = tk.Label(root, text="Sepal Length:")
label1.pack()
entry1 = tk.Entry(root)
entry1.pack()

label2 = tk.Label(root, text="Sepal Width:")
label2.pack()
entry2 = tk.Entry(root)
entry2.pack()

#define the input

label3 = tk.Label(root, text="petal Length:")
label3.pack()
entry3 = tk.Entry(root)
entry3.pack()

label3 = tk.Label(root, text="petal Width:")
label3.pack()
entry3 = tk.Entry(root)
entry3.pack()

# Define the prediction function
def predict():
    sepal_length = float(entry1.get())
    sepal_width = float(entry2.get())
     
    test_data = sc.transform([[sepal_length, sepal_width]])
    prediction = classifier.predict(test_data)
    result_label.config(text="Predicted Species: " + prediction[0])
    

def predict():
    petal_length=float(entry3.get())
    petal_width=float(entry3.get())
    test_data = sc.transform([[petal_length,petal_width]])
    prediction = classifier.predict(test_data)
    result_label.config(text="Predicted Species: " + prediction[0])

# Define the prediction button
button = tk.Button(root, text="predict", command=predict)
button.pack()

# Define the result label
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()






















