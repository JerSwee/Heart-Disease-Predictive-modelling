#Reference: https://medium.com/@allwindicaprio/plotting-graphs-using-python-and-matplotlib-f55c9b99c338
#https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/

import matplotlib.pyplot as plt

#example1 - Plotting a line
# x = [1, 2, 3]
# y = [2, 4, 1]
#
# plt.plot(x, y)
#
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
#
# plt.title('Line graph!')
#
# plt.show()

#exmaple2- Plotting two or more lines on the same plot
# x = ['Maths', 'Physics', 'Chemistry']
#
# y1 = [95, 88, 45]
#
#
# plt.plot(x, y1, label="John")
#
# y2 = [67, 45, 56]
#
#
# plt.plot(x, y2, label="David")
#
#
# y3 = [28, 67, 90]
#
# plt.plot(x, y3, label="Tom")
#
# plt.xlabel('Subjects')
# plt.ylabel('Marks')
#
# plt.title('Three lines on same graph!')
#
# plt.legend()
#
# plt.show()

#Example3 - Scatter plot
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [2, 4, 5, 7, 6, 8, 9, 11, 12, 12]
#
# plt.scatter(x, y, label="stars", color="green",
#             marker="1", s=30)
#
#
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
#
# plt.title('Scatter plot')
# plt.legend()
#
# plt.show()

#Example4 - Pie-chart
# items = ['Samsung', 'Huawei', 'Apple', 'Oppo']
#
# proportions = [40, 20, 30, 50]
#
# colors = ['r', 'y', 'g', 'b']
#
# plt.pie(proportions, labels=items, colors=colors,
#         startangle=20, shadow=True, explode=(0.1, 0, 0, 0),
#         radius=1.2, autopct='%1.1f%%')
#
#
# plt.title('Market share of smart phones')
# plt.legend()
# plt.show()

#Example5 - Bar Chart
# x_units = [1, 2, 3, 5]
#
# y_units = [10, 24, 36, 48]
#
# tick_label = ['one', 'two', 'three', 'five']
#
# plt.bar(x_units, y_units, tick_label=tick_label,
#         width=0.8, color=['red', 'green'])
#
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.title('My bar chart!')
#
# plt.show()

#Example6 - # Histogram
# frequencies
# ages = [2, 5, 70, 40, 30, 45, 50, 45, 43, 40, 44,
#         60, 7, 13, 57, 18, 90, 77, 32, 21, 20, 40]
#
# range = (0, 100)
# bins = 20
#
# # histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, optional
# plt.hist(ages, bins, range, color='blue',
#          histtype='bar', rwidth=0.5)
#
# plt.xlabel('age')
# plt.ylabel('No. of people')
# plt.title('Ages of people')
#
# plt.show()

#Example7 - Customization of graphs
# # x axis values
# x = [1, 2, 3, 4, 5, 6]
# # corresponding y axis values
# y = [2, 4, 1, 5, 2, 6]
#
# # plotting the points
# plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
#          marker='o', markerfacecolor='blue', markersize=12)
#
# # setting x and y axis range
# plt.ylim(1, 8)
# plt.xlim(1, 8)
#
# # naming the x axis
# plt.xlabel('x - axis')
# # naming the y axis
# plt.ylabel('y - axis')
#
# # giving a title to my graph
# plt.title('Some cool customizations!')
#
# # function to show the plot
# plt.show()

#Example 8 - Plotting curves of given equation
import numpy as np

# setting the x - coordinates
x = np.arange(0, 2 * (np.pi), 0.1)
# setting the corresponding y - coordinates
y = np.sin(x)

# potting the points
plt.plot(x, y)

# function to show the plot
plt.show()