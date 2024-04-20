# Importing libraries ---------------------------------------
import numpy as np  # for math functions
# sklearn for data analysis & statistics
from sklearn import datasets, linear_model, model_selection
import matplotlib.pyplot as plt  # for plotting

# Loading the database
X, y = datasets.load_diabetes(return_X_y=True)

print(len(X))

# Sort the matrix for processing
X = X[:, np.newaxis, 2]

print(len(X))

# Create a training model by splitting to train/test values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

# Define the used model (here it is Linear Regression)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Do the prediction
y_pred = model.predict(X_test)

# Plotting function
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.xlabel("Scaled BMIs")
plt.ylabel("Disease Progression")
plt.title("A Graph Plot Showing Diabetes Progression Against BMI")
plt.show()

# IMPORTING TO PGFPLOTS -----
import pandas as pd

data = {
    'xtest': np.array(X_test).flatten(),
    'ytest': np.array(y_test).flatten(),
    'yred': np.array(y_pred).flatten()
}
df = pd.DataFrame(data)
df.to_csv("Regression.csv", index=False, header=False)