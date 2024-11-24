from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

features = iris.data
lables = iris.target

sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))

petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

clf = KNeighborsClassifier()
clf.fit(features, lables)

pred = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

print(pred)

pred = pred.item()

if pred == 0:
    print("Iris-Setosa")
elif pred == 1:
     print("Iris-Versicolour")
elif pred == 2:
     print("Iris-Virginica")