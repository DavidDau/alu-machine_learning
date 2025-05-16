#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 8]
cov = [[15, 8], [8, 15]]  # Fixed variable name from 'vov' to 'cov'

np.random.seed(5)

x, y = np.random.multivariate_normal(mean, cov, 2000).T  # Fixed syntax error in x,y assignment
y += 180

plt.scatter(x, y, color="magenta")
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")
plt.show()
