from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
# splitting into features and labels
x = iris.data
y = iris.target

print(x.shape)  # (150, 4) -- 150 instances and 4 features
print(y.shape)  # (150, 1) -- 150 instances and 1 label

# does the training and testing split for you;
# does 20% allocation for testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(X_train.shape)  # (120, 4) -- 120 instances and 4 features
print(X_test.shape)  # (30, 4) -- 30 instances and 4 features
print(y_train.shape)  # (120, 1) -- 120 instances and 1 label
print(y_test.shape)  # (30, 1) -- 30 instances and 1 label
