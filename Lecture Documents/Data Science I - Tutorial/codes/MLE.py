# Importing libraries ---------------------------------------
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Retrieve the boston data for analysis
data_url = "http://lib.stat.cmu.edu/datasets/boston"
# Downloads the data and puts it into a dataframe
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=1)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10,
            label='Train data')

# plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10,
            label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()


dfA = pd.DataFrame(reg.predict(X_train) - y_train, reg.predict(X_train))
print(dfA)
dfA.to_csv("MLE_train.csv",header=False)

dfB = pd.DataFrame(reg.predict(X_test) - y_test, reg.predict(X_test))
print(dfB)
dfB.to_csv("MLE_test.csv",header=False)
