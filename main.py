import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


"""
best = 0
for _ in range(100):
    linear = LinearRegression()

    linear.fit(X_train, y_train)
    acc = linear.score(X_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmathmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

"""



pickle_in = open("studentmathmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n',  linear.coef_)
print("Intercept: \n" , linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()