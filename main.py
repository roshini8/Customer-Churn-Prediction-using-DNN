import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Churn_Modelling.csv")
X = data.iloc[:, 3:-1].values  # credit socre -> estimated salary

# Target [Exited]
Y = data.iloc[:, -1].values

LE1 = LabelEncoder()
X[:, 2] = np.array(LE1.fit_transform(X[:, 2]))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def build_ann_model(units):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=units, activation="relu"))  # Layer 1
    ann.add(tf.keras.layers.Dense(units=units, activation="relu"))  # Layer 2
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))  # Layer 3
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return ann

neurons = [5, 10]
accuracies = []

for num_neurons in neurons:
    ann = build_ann_model(num_neurons)
    ann.fit(X_train, Y_train, batch_size=16, epochs=100)
    predictions = ann.predict(X_test)
    predictions = (predictions > 0.5)
    print(f"Predictions (Neurons={num_neurons}):\n{predictions}")
    accuracy = accuracy_score(Y_test, predictions)
    accuracies.append(accuracy)
    print(f"Accuracy (Neurons={num_neurons}): {accuracy}")

    cm = confusion_matrix(Y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Neurons={num_neurons})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    report = classification_report(Y_test, predictions)
    print(f"Classification Report (Neurons={num_neurons}):\n{report}")

print('\nComparison')
for i, num_neurons in enumerate(neurons):
    print(f"Accuracy (Neurons={num_neurons}): {accuracies[i]}")



