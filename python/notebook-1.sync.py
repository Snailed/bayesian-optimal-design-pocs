import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
# %%
d = 2
np.random.seed(1337)
x = np.random.rand(100) * 5
y = np.random.normal(loc = 0, scale=0.3, size=100) + x
X = np.dstack((x,y))[0]
# %%
plt.scatter(X[:,0], X[:,1])
# %%
# Start cov
covariance = np.zeros((d,d))
mean = 0
theta_prior = np.random.normal(loc=mean, scale=1.0, size=[d])
def f(x):
    x_reshaped = np.reshape(x, (x.shape[0], 1))
    zeros = np.zeros((x.shape[0], 1))
    x_aug = np.concatenate((x_reshaped, zeros), axis=1)
    print(theta_prior)
    return theta_prior.T * x_aug
plt.scatter(X[:,0], X[:,1])
print(f(np.sort(x)))
plt.plot(np.sort(x), f(np.sort(x)))
