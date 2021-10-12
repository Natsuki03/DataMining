import dataset1
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#ex1.7

path = "/Users/e195767/VS code/Data_Mining_Code/exam/"
data = dataset1.load_tsv_from_df(path, "dataset1.tsv")


#ex1.8 & ex1.9

def learn(dataset, test_size, random_state=0):
    feature = dataset["観測値"]
    target = dataset["真値"]
    X_train, y_train, X_test, y_test = train_test_split(feature, target, random_state=random_state, test_size=test_size)

    X_train = np.array(X_train).reshape(1, -1)
    y_train = np.array(y_train).reshape(1, -1)
    X_test = np.array(X_test).reshape(1, -1)
    y_test = np.array(y_test).reshape(1, -1)

    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    return lr, X_train, y_train, X_test, y_test

lr, X_train, y_train, X_test, y_test = learn(data, test_size=0.2)

#ex1.10

predicted = lr.predict(X_test)
print(predicted)