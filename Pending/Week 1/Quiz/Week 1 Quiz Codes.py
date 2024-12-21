# 1. Mean of datasets
 
# Q.6 ---------------------------------

import numpy as np

def reshape(x):
  """return x_reshaped as a flattened vector of the multi-dimensional array x"""
  x_reshaped = ???
  return x_reshaped

##################################################################################

# 3. Covariance matrix of a two-dimensional dataset

# Q.1 ---------------------------------

# RUN THE CODE ONCE, THEN UNCOMMENT LINE 29 TO VISUALISE COVARIANCE

fig, ax = plt.subplots()

#Choose an array by deleting the # in front of the word "data" below. 
#To switch, put the # back and delete another one

#Random: 
data = np.array([[1,2],[5,4],[-2,-3],[4,-2],[2,3],[8,-9]])

#Straight line: 
#data = np.array([[1,1],[-3,-3],[2,2],[7,7]])

#Q1: square
#data = np.array([[0,0],[4,4],[0,4],[4,0]])

#Feel free to input your own array or modify the ones above! 

# First calculate the mean with NumPy function np.mean(). 
# The first argument is the dataset and "axis" specifies the direction
# Variance in 1D can be calculated similarly with np.var()
mean_data = np.mean(data, axis=0)
create_plot(data) #which also adds 1d variances

area=0
mean = mean_data

for i in range(len(data)):
#    show_rectangle(mean, data[i])
    # and a calculation that adds (or subtracts) 
    # the value of the area to our value of the covariance:
    area += calculate_area(mean, data[i])

plt.show()

# Q.3 ---------------------------------

data = np.array([[1,2],[5,4]])
#data *= 2
#Uncomment the line above to multiply by 2 and run again

mean_data = np.mean(data, axis=0)
create_plot(data)

area=0
mean = mean_data

for i in range(len(data)):
    show_rectangle(mean, data[i])
    area += calculate_area(mean, data[i])

plt.show()

# Q.4 ---------------------------------

data = np.array([[1,2],[5,4]])
#data += [2,2]
#Uncomment line above after first run to add [2,2], then run again

mean_data = np.mean(data, axis=0)
create_plot(data)

area=0
mean = mean_data

for i in range(len(data)):
    show_rectangle(mean, data[i])
    area += calculate_area(mean, data[i])

plt.show()