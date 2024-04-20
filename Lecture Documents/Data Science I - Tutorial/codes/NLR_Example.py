# Importing libraries ---------------------------------------
import pandas as pd  # import for reading .csv
import numpy as np  # import for math function
import matplotlib.pyplot as plt  # import for plotting
from sklearn.metrics import mean_absolute_error, r2_score

# Read the CSV file
df = pd.read_csv("GDP.csv")

# Display the first few rows of the dataframe
print(df.head())

plt.figure(figsize=(8, 5))
x_original, y_original = df["Year"].values, df["Value"].values
plt.plot(x_original, y_original, 'bo')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('Original GDP Data')
plt.show()

# Plot a simple logistic model curve
X_logistic = np.arange(-5.0, 5.0, 0.1)
Y_logistic = 1.0 / (1.0 + np.exp(-X_logistic))

# Orange color for the logistic curve
plt.plot(X_logistic, Y_logistic, color='green')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.title('Simple Logistic Model Curve')
plt.show()


# Define the sigmoid function
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


# Set initial values for logistic function parameters
beta_1_initial = 0.10
beta_2_initial = 1990.0

# Apply logistic function to the data
Y_pred_logistic = sigmoid(x_original, beta_1_initial, beta_2_initial)

plt.plot(x_original, Y_pred_logistic * 15000000000000., color='purple', label='Initial Prediction')
plt.plot(x_original, y_original, 'go', label='Data')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.legend()
plt.title('Initial Logistic Regression Fit')
plt.show()

# Normalize the data
x_normalized = x_original / max(x_original)
y_normalized = y_original / max(y_original)

from scipy.optimize import curve_fit

# Fit the sigmoid function to the normalized data
popt, pcov = curve_fit(sigmoid, x_normalized, y_normalized)

# Print the final parameters
print("Beta_1 = %f, Beta_2 = %f" % (popt[0], popt[1]))

# Create a new x range for the fitted sigmoid curve
x_fit = np.linspace(1960, 2015, 55) / max(x_original)

# Apply the sigmoid function with the fitted parameters
y_fit = sigmoid(x_fit, *popt)

# Plot the normalized data, the sigmoid fit, and the legend
plt.figure(figsize=(8, 5))
plt.plot(x_normalized, y_normalized, 'go', label='Normalized Data')  # Changed color to green for data points
plt.plot(x_fit, y_fit, linewidth=3.0, color='purple',
         label='Sigmoid Fit')  # Changed color to purple for the sigmoid fit line
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('Normalized Sigmoid Regression Fit')
plt.show()

# Split data into train/test sets
random_mask = np.random.rand(len(df)) < 0.8
train_x = x_normalized[random_mask]
test_x = x_normalized[~random_mask]
train_y = y_normalized[random_mask]
test_y = y_normalized[~random_mask]

# Build the model using the train set
popt_train, pcov_train = curve_fit(sigmoid, train_x, train_y)

# Predict using the test set
y_hat_test = sigmoid(test_x, *popt_train)

# Evaluate the model
mae = mean_absolute_error(test_y, y_hat_test)
mse = np.mean((y_hat_test - test_y) ** 2)
r2 = r2_score(y_hat_test, test_y)

# Print the evaluation metrics
print("Mean Absolute Error: %.2f" % mae)
print("Mean Squared Error: %.2f" % mse)
print("R2-score: %.2f" % r2)

data = {
    'X': np.array(x_fit).flatten(),
    'Y': np.array(y_fit).flatten(),
    'Z': np.array(x_normalized).flatten(),

}
print(data)
df = pd.DataFrame(data)
print(df)

df.to_csv("NLR.csv", index=False, header=False)

