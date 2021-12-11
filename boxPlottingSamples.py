#References: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

#Example1 - Creating Box Plot
# # Creating dataset
# np.random.seed(10)
# data = np.random.normal(100, 20, 200)
#
# fig = plt.figure(figsize=(10, 7))
#
# # Creating plot
# plt.boxplot(data)
#
# # show plot
# plt.show()

#Example2 - Customizing Box Plot
# Creating dataset
np.random.seed(10)

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]

fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])

# Creating plot
bp = ax.boxplot(data)

# show plot
plt.show()