class RecursiveLeastSquares(object):

    # x0 - initial estimate used to initialize the estimator
    # P0 - initial estimation error covariance matrix
    # R  - covariance matrix of the measurement noise
    def __init__(self, x0, P0, R):
        # initialize the values
        self.x0 = x0
        self.P0 = P0
        self.R = R

        # this variable is used to track the current time step k of the estimator
        # after every time step arrives, this variables increases for one
        # in this way, we can track the number of variblaes
        self.currentTimeStep = 0

        # this list is used to store the estimates xk starting from the initial estimate
        self.estimates = []
        self.estimates.append(x0)

        # this list is used to store the estimation error covariance matrices Pk
        self.estimationErrorCovarianceMatrices = []
        self.estimationErrorCovarianceMatrices.append(P0)

        # this list is used to store the gain matrices Kk
        self.gainMatrices = []

        # this list is used to store estimation error vectors
        self.errors = []

    # this function takes the current measurement and the current measurement matrix C
    # and computes the estimation error covariance matrix, updates the estimate,
    # computes the gain matrix, and the estimation error
    # it fills the lists self.estimates, self.estimationErrorCovarianceMatrices, self.gainMatrices, and self.errors
    # it also increments the variable currentTimeStep for 1

    # measurementValue - measurement obtained at the time instant k
    # C - measurement matrix at the time instant k

    def predict(self, measurementValue, C):
        import numpy as np

        # compute the L matrix and its inverse, see Eq. 43
        Lmatrix = self.R + np.matmul(C, np.matmul(self.estimationErrorCovarianceMatrices[self.currentTimeStep], C.T))
        LmatrixInv = np.linalg.inv(Lmatrix)
        # compute the gain matrix, see Eq. 42 or Eq. 48
        gainMatrix = np.matmul(self.estimationErrorCovarianceMatrices[self.currentTimeStep], np.matmul(C.T, LmatrixInv))

        # compute the estimation error
        error = measurementValue - np.matmul(C, self.estimates[self.currentTimeStep])
        # compute the estimate, see Eq. 49
        estimate = self.estimates[self.currentTimeStep] + np.matmul(gainMatrix, error)

        # propagate the estimation error covariance matrix, see Eq. 50
        ImKc = np.eye(np.size(self.x0), np.size(self.x0)) - np.matmul(gainMatrix, C)
        estimationErrorCovarianceMatrix = np.matmul(ImKc, self.estimationErrorCovarianceMatrices[self.currentTimeStep])

        # add computed elements to the list
        self.estimates.append(estimate)
        self.estimationErrorCovarianceMatrices.append(estimationErrorCovarianceMatrix)
        self.gainMatrices.append(gainMatrix)
        self.errors.append(error)

        # increment the current time step
        self.currentTimeStep = self.currentTimeStep + 1


import numpy as np
import matplotlib.pyplot as plt

# define the true parameters that we want to estimate

# true value of the parameters that will be estimated
initialPosition = 100
acceleration = 1
initialVelocity = 2

# noise standard deviation
noiseStd = 1;

# simulation time
simulationTime = np.linspace(0, 15, 1000)
# vector used to store the somulated position
position = np.zeros(np.size(simulationTime))

# simulate the system behavior
for i in np.arange(np.size(simulationTime)):
    position[i] = initialPosition + initialVelocity * simulationTime[i] + (acceleration * simulationTime[i] ** 2) / 2

# add the measurement noise
positionNoisy = position + noiseStd * np.random.randn(np.size(simulationTime))

# verify the position vector by plotting the results
plotStep = 300
plt.plot(simulationTime[0:plotStep], position[0:plotStep], linewidth=4, label='Ideal position')
plt.plot(simulationTime[0:plotStep], positionNoisy[0:plotStep], 'r', label='Observed position')
plt.xlabel('time')
plt.ylabel('position')
plt.legend()
plt.savefig('data.png', dpi=300)
plt.show()

x0 = np.random.randn(3, 1)
P0 = 100 * np.eye(3, 3)
R = 0.5 * np.eye(1, 1)

# create a recursive least squares object
RLS = RecursiveLeastSquares(x0, P0, R)

# simulate online prediction
for j in np.arange(np.size(simulationTime)):
    C = np.array([[1, simulationTime[j], (simulationTime[j] ** 2) / 2]])
    RLS.predict(positionNoisy[j], C)

# extract the estimates in order to plot the results
estimate1, estimate2, estimate3 = [], [], []

for j in np.arange(np.size(simulationTime)):
    estimate1.append(RLS.estimates[j][0])
    estimate2.append(RLS.estimates[j][1])
    estimate3.append(RLS.estimates[j][2])

# create vectors corresponding to the true values in order to plot the results
estimate1true = initialPosition * np.ones(np.size(simulationTime))
estimate2true = initialVelocity * np.ones(np.size(simulationTime))
estimate3true = acceleration * np.ones(np.size(simulationTime))

# plot the results
steps = np.arange(np.size(simulationTime))
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# First plot - Position
ax[0].plot(steps, estimate1true, color='red', linestyle='-', linewidth=6,
           label='True value of position')
ax[0].plot(steps, estimate1, color='blue', linestyle='-', linewidth=3,
           label='True value of position')
ax[0].set_xlabel("Discrete-time steps k", fontsize=14)
ax[0].set_ylabel("Position", fontsize=14)
ax[0].tick_params(axis='both', labelsize=12)
ax[0].set_ylim(98, 102)
ax[0].grid(), ax[0].legend(fontsize=14)

# Second plot - Velocity
ax[1].plot(steps, estimate2true, color='red', linestyle='-', linewidth=6,
           label='True value of velocity')
ax[1].plot(steps, estimate2, color='blue', linestyle='-', linewidth=3,
           label='Estimate of velocity')
ax[1].set_xlabel("Discrete-time steps k", fontsize=14)
ax[1].set_ylabel("Velocity", fontsize=14)
ax[1].tick_params(axis='both', labelsize=12)
ax[1].grid(), ax[1].legend(fontsize=14)

# Third plot - Acceleration
ax[2].plot(steps, estimate3true, color='red', linestyle='-', linewidth=6,
           label="True value of acceleration")
ax[2].plot(steps, estimate3, color='blue', linestyle='-', linewidth=3,
           label="Estimate of acceleration")
ax[2].set_xlabel("Discrete-time steps k", fontsize=14)
ax[2].set_ylabel("Acceleration", fontsize=14)
ax[2].tick_params(axis='both', labelsize=12)
ax[2].grid(), ax[2].legend(fontsize=14)
plt.show()


import pandas as pd

save1df = pd.DataFrame(simulationTime[0:plotStep], position[0:plotStep])
save1df.to_csv("RLE_Ideal.csv", header=False)
save2df = pd.DataFrame(simulationTime[0:plotStep],
                       positionNoisy[0:plotStep])
save2df.to_csv("RLE_Observed.csv", header=False)

data = {
    'steps': steps,
    'estimate': np.array(estimate3).flatten(),
    'true': estimate3true
}


save3df = pd.DataFrame(data)
save3df.to_csv("RLE_3.csv", index=False, header=False)
