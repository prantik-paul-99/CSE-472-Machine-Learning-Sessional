import numpy as np

n = int(input("Enter the dimension of matrix: "))

A = np.random.randint(-100, 100, (n,n))
while np.linalg.det(A) == 0:
    A = np.random.randint(-100, 100, (n,n))

print("Original matrix is: " ,A)

#eigen decomposition
eig_vals, eig_vects = np.linalg.eig(A)

print("Eigen values are: ", eig_vals)
print("Eigen vectors are: ", eig_vects)

#Reconstruction
A_recon = np.dot(np.diag(eig_vals), np.linalg.inv(eig_vects))
A_recon = np.dot(eig_vects, A_recon)

print("Reconstructed matrix is: ", A_recon)

print("Is the reconstruction perfect? ")

if(np.allclose(A, A_recon)):
    print("yes")
else:
    print("no")
