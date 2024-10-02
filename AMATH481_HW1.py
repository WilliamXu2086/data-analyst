import numpy as np # type: ignore

# Question 1
# Newton-Raphson Method
x = np.array([-1.6]) # initial guess
times_nr = 0
for j in range(1000):
    x = np.append(
        x, x[j] - (x[j]*np.sin(3*x[j]) - np.exp(x[j])) 
        / (np.sin(3*x[j]) + 3*x[j]*np.cos(3*x[j]) - np.exp(x[j]))
    )
    fc = x[j+1]*np.sin(3*x[j+1]) - np.exp(x[j+1])
    times_nr += 1
    if abs(fc) < 1e-6:
        break
A1 = x

# Bisection Method
A2 = np.array([])
times_b = 0
xr = -2.8; xl = -4 # initial guess
for j in range(0, 100):
    times_b += 1
    xc = (xr + xl)/2
    A2 = np.append(A2, xc)
    fc = np.exp(xc) - np.tan(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc
    if abs(fc) < 1e-6: 
        break
A3 = np.array([times_nr,times_b])


# Question 2
# A - z
A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([[1],[0]])
y = np.array([[0],[1]])
z = np.array([[1],[2],[-1]])

# A4 - A12
A4 = A + B
A5 = 3*A - 4*B
A6 = np.matmul(A,x)
A7 = np.matmul(B,(x - y))
A8 = np.matmul(D,x)
A9 = np.matmul(D,y) + z
A10 = np.matmul(A,B)
A11 = np.matmul(B,C)
A12 = np.matmul(C,D)

print(A1)
print(A2)
print(A3)
print(A4)
print(A5)
print(A6)
print(A7)
print(A8)
print(A9)
print(A10)
print(A11)
print(A12)