# Importing libraries ---------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Retrieving the csv data for polyfit
df = pd.read_csv(
    'https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish.csv'
)

X, y = df.iloc[:, 1:2].values, df.iloc[:, 2].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly_reg2 = PolynomialFeatures(degree=2)
poly_reg3 = PolynomialFeatures(degree=3)

X_poly, X_poly3 = poly_reg2.fit_transform(X), poly_reg3.fit_transform(X)
lin_reg_2, lin_reg_3 = LinearRegression(), LinearRegression()
lin_reg_2.fit(X_poly, y), lin_reg_3.fit(X_poly3, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='green')
plt.title('Simple Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg2.fit_transform(X)), color='green')
plt.plot(X, lin_reg_3.predict(poly_reg3.fit_transform(X)), color='yellow')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)  # This will give us a vector.We will have to convert this into a matrix
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_3.predict(poly_reg3.fit_transform(X_grid)), color='lightgreen')
plt.show()


data = {
    'X': np.array(X).flatten(),
    'y': np.array(y).flatten(),
    "p1": np.array(lin_reg.predict(X)).flatten(),
    "xgrid": np.array(X_grid).flatten(),
    "p2": np.array(lin_reg_2.predict(poly_reg2.fit_transform(X_grid))).flatten(),
    "p3": np.array(lin_reg_3.predict(poly_reg3.fit_transform(X_grid))).flatten(),
}
print(data)
df = pd.DataFrame(data)
print(df)

df.to_csv("PLE.csv", index=False, header=False)