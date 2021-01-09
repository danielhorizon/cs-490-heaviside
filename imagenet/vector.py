import torch 
import numpy as np

# want to test out computing something along an axis 

a = torch.Tensor([
    [1, 2, 3, 4],
    [4, 5, 6, 7],
    [7, 8, 9, 10]
])

b = torch.Tensor([
    [2, 3, 4, 5],
    [6, 7, 8, 9],
    [10, 11, 12, 13]
])

'''
- Perform an operation on each column of the Tensor
- Return the sum of the column
'''

# Takes in both columns, adds them together, and then returns a single column 
def double_vals(x, y): 
    return x + y 

def vector_operation(X, Y): 
    X = X.numpy()
    Y = Y.numpy()
    vfunc = np.vectorize(double_vals)

    return vfunc(X, Y)


print(vector_operation(a,b))
