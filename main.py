import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("car.data")

# prints a summary of top 5 rows along with headers
print(data.head())

# "buying", "maintenance" and "safety" as features
X = data[["buying", "maint", "safety"]].values
# "class" as label
y = data[["class"]]

# Encoding Features
# string data is not useful, to be converted to int
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

# Encoding Labels
label_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
y["class"] = y["class"].map(label_mapping)
y = np.array(y)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# building the model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights="uniform")
knn.fit(X_train, y_train)
# checking accuracy of predictions
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

# sample run
val = 20
print("Car: ", X[val])
print("prediction value:", knn.predict(X)[val], "(actual value: ", y[val], ")")
