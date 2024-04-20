# Importing libraries ---------------------------------------
import numpy as np  # for math functions
import matplotlib.pyplot as plt  # for plotting


def estimateCoefficient(x, y):
    # number of observations/points
    n = np.size(x)
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return b_0, b_1


def plotRegressionLine(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m", marker="o", s=30)
    # predicted response vector
    y_pred = b[0] + b[1] * x
    # plotting the regression line
    plt.plot(x, y_pred, color="g")
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# observations / data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# estimating coefficients
b = estimateCoefficient(x, y)
plotRegressionLine(x, y, b)
print("Estimated coefficients:")
print("b_0: ", b[0])
print("b_1: ", b[1])
