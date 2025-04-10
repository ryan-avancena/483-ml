import numpy as np

x1 = np.array([0,2,4,6])
x2 = np.array([1,3,8,10])
y = np.array([3,5,9,10])

X = np.column_stack((np.ones_like(x1), x1, x2))

# Print X
print("Design Matrix X:\n", X)
XT_X = X.T @ X  # Matrix multiplication
print("X^T X:\n", XT_X)

XT_X = X.T @ X  # Matrix multiplication
print("X^T X:\n", XT_X)


# Compute the inverse of (X^T X)
XT_X_inv = np.linalg.inv(XT_X)
print("Inverse of X^T X:\n", XT_X_inv)

XT_Y = X.T @ y  # Matrix multiplication
print("X^T Y:\n", XT_Y)

# Compute the coefficients
beta = XT_X_inv @ XT_Y

# Print the coefficients
print(f"Intercept (β0): {beta[0]:.4f}")
print(f"Coefficient for x1 (β1): {beta[1]:.4f}")
print(f"Coefficient for x2 (β2): {beta[2]:.4f}")
