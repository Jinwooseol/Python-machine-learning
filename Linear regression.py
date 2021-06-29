import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head()) #show data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict],1))
y = np.array(data[predict]) # labels

# x_train -> x axis, y_train -> y axis, x_test -> x의 accuracy, y_test same
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # test_size means accuracy percentage 10%

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # test_size means accuracy percentage 10%
    linear = sklearn.linear_model.LinearRegression()

    linear.fit(x_train, y_train)    # x_train, y_train data에 가장 잘 맞는 best linear line
    acc = linear.score(x_test, y_test)  # accuracy of linear regression

    print('Accuracy: ',acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:        # Save the model using pickle
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")       # Load the saved model
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'absences'        # x axis (using dynamically)
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])  # scatter(x axis, y axis)
pyplot.xlabel(p)                # x axis label under plot
pyplot.ylabel("Final grade")    # y axis label
pyplot.show()