# Importing libraries ---------------------------------------
import pandas as pd  # import to create a database for the values
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# create DataFrame
df = pd.DataFrame({'hours': [1, 2, 4, 5, 5, 6, 6, 7,
                             8, 10, 11, 11, 12, 12, 14],
                   'score': [64, 66, 76, 73, 74, 81, 83,
                             82, 80, 88, 84, 82, 91, 93, 89]})
# print the DataFrame to the terminal outpu

# define predictor and response variables
x, y = df['hours'], df['score']
print(x)
# add constant to predictor variables
x = sm.add_constant(x)
print(x)
# fit linear regression model
model = sm.OLS(y, x).fit()

# view model summary
print(model.summary())

# find line of best fit
a, b = np.polyfit(df['hours'], df['score'], 1)


# Plotting function
plt.scatter(df['hours'], df['score'], color='purple')
plt.plot(df['hours'], a * df['hours'] + b)
plt.text(1, 90, 'y = ' + '{:.3f}'.format(b) + ' + {:.3f}'.format(a) + 'x', size=12)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

# IMPORTING TO PGFPLOTS ----
df.to_csv("OLS_Example.csv", index=False, header=False)
savedf = pd.DataFrame(df['hours'], a * df['hours'] + b)
print(savedf)
savedf.to_csv("OLS_Example_Fit.csv",header=False)