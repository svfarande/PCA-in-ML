# 1. Dot product
 
# Q.4 ---------------------------------

import numpy as np

def length(x):
  """Compute the length of a vector"""
  length_x = -1 # <--- compute the length of a vector x here.
  
  return length_x
  
print(length(np.array([1,0]))) 

##################################################################################

# 2. Properties of inner products

# Q.5 ---------------------------------

import numpy as np

def dot(a, b):
  """Compute dot product between a and b.
  Args:
    a, b: (2,) ndarray as R^2 vectors
  
  Returns:
    a number which is the dot product between a, b
  """
  
  dot_product = ???
  
  return dot_product

# Test your code before you submit.
a = np.array([1,0])
b = np.array([0,1])
print(dot(a,b))

##################################################################################

# Angles between vectors using a non-standard inner product

# Q.2 ---------------------------------

# the matrix A defines the inner product
A = np.array([[1, -1/2],[-1/2,5]])
x = np.array([0,-1])
y = np.array([1,1])

def find_angle(A, x, y):
    """Compute the angle"""
    inner_prod = x.T @ A @ y
    # Fill in the expression for norm_x and norm_y below
    norm_x = 
    norm_y =
    alpha = inner_prod/(norm_x*norm_y)
    angle = np.arccos(alpha)
    return np.round(angle,2) 

find_angle(A, x, y)

# Q.4 ---------------------------------

# Fill in the arrays and use the function `find_angle` defined for you to aid in your calculation
A = np.array()
x = np.array()
y = np.array()

find_angle(A, x, y)

# Q.5 ---------------------------------

# Fill in the following arrays and use `find_angle` to aim your calculation.
A = np.array()
x = np.array()
y = np.array()

find_angle(A, x, y)