#!usr/bin/venv python3
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)
plt.plot(x, y1,label = "C-14", color = "red", linestyle = "dashed")
plt.plot(x, y2, label = "Ra-226", color = "green", linestyle = "solid")
plt.xlim(0,20000)
plt.ylim(0,1)
plt.title("Exponential Decay of Radiactive Elements")
plt.xlabel("Time (years)")
plt.ylabel("Fractional Remaining")
plt.legend(loc = "upper right")
plt.show()

