import numpy as np


n = int(input("Enter the dimension n of matrix: "))
m = int(input("Enter the dimension m of matrix: "))

A = np.random.randint(-100., 100., (n,m))

print("Original matrix is: ", A)

#singular value decomposition
U, D, V = np.linalg.svd(A, full_matrices=True)

D = np.diag(D)

zeros = np.zeros((n,m))
zeros[:D.shape[0], :D.shape[1]] = D

D = zeros.T

#print(s)

D[D != 0.] = 1 / D[D != 0.]
#D = np.reciprocal(D, where= D!=0.0)
#np.reciprocal(D.data, out = D.data)

#print(np.shape(U))
#print(np.shape(D))
#print(np.shape(V))

#print(V.T)
#print(U.T)
#print(D)

A_inv_builtin = np.linalg.pinv(A)

#A_inv_builtin = np.reciprocal(A_inv_builtin, where= A_inv_builtin!=0)

print("Pseudo-Inverse matrix (Built in Function) is: ", A_inv_builtin)

#using equation
A_inv_svd = np.dot(V.T, np.dot(D, U.T))

#A_inv_svd = np.reciprocal(A_inv_svd, where= A_inv_svd!=0)

print("Pseudo-Inverse matrix (SVD) is: ", A_inv_svd)

#Check if both are equal
print("Are both equal?")

if(np.allclose(A_inv_builtin, A_inv_svd)):
    print("yes")
else:
    print("no")
