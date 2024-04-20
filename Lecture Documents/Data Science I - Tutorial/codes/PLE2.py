import operator
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 0.4, 500)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(0, 1, 500)
print(y)
# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]


polynomial_features = PolynomialFeatures(degree=1)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
r2 = r2_score(y, y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()



import pandas as pd

data = {
    'X': np.array(x).flatten(),
    'y': np.array(y_poly_pred).flatten(),
    "z": np.array(y).flatten(),
}
df = pd.DataFrame(data)

df.to_csv("PLE2.csv", index=False, header=False)