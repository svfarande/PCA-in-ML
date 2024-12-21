# 2. Steps of PCA

# Q.1 ---------------------------------

plt.plot(x, y, 'o',markersize=10)
m = 3        #Change this value
plot_line(m)

# Q.5 ---------------------------------
# Change the line below to subtract the mean from the data
# You can use NumPy functions to help you compute the results

# `data` is a 2D NumPy array with shape (N, D)
# where N is the number of samples and D is the dimensionality.
# You should change the right hand side of the statement below to compute the
# normalized dataset.
data_normalized = data
plot(data_normalized)

# Copy your code from above to calculate data_normalized correctly
data_normalized = data 

# Q.7 ---------------------------------
# Complete the code below to compute cov[x, x] cov[y, x] and cov[y, y].
# cov_xy has been provided as an example 
cov_xx = ??
cov_xy = (1/N)*np.sum(data_normalized[:,0]*data_normalized[:,1])
cov_yx = ??
cov_yy = ??

# This prints the final covariance matrix
# using the elements computed above
# Once you have finished writing the code above, 
# DIRECTLY copy the output of the following function
# to the answer cell.
print_covariance_matrix() 

# Q.8 ---------------------------------
plot_projected_coords()