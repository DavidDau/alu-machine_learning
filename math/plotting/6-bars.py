#!/usr/bin/env python3
"""Module for creating a stacked bar chart of fruit distribution"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(5)

# Generate random fruit data
fruit = np.random.randint(0, 20, (4,3))

# Define labels and colors
fruit_labels = ["apples", "bananas", "oranges", "peaches"]
colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
people_labels = ["Farrah", "Fred", "Felicia"]

# Create figure and axis objects
fig, ax = plt.subplots()

# Initialize bottom of stacked bars
bottom = np.zeros(3)

# Create stacked bars
for i, fruit_type in enumerate(fruit_labels):
    fruit_data = fruit[i,:]
    ax.bar(np.arange(3), fruit_data, bottom=bottom, width=0.5, 
           label=fruit_type, color=colors[i])
    bottom += fruit_data

# Customize plot
ax.set_xticks(np.arange(3))
ax.set_xticklabels(people_labels)
ax.set_ylabel("Quantity of Fruit")
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_title("Number of Fruits per Person")
ax.legend(loc="upper right")

plt.show()
