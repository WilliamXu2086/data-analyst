import numpy as np # type: ignore

# Question 1
# Newton-Raphson Method
x = np.array([-1.6]) # initial guess
times_nr = 0
for j in range(1000):
    x = np.append(x, x[j] - ((x[j] * np.sin(3*x[j]) - np.exp(x[j]))
                /(np.sin(3*x[j]) + 3*x[j]*np.cos(3*x[j]) - np.exp(x[j])))
    )
    times_nr += 1
    fc = x[j] * np.sin(3*x[j]) - np.exp(x[j])
    if abs(fc) < 1e-6:
        break
A1 = x

# Bisection Method
A2 = np.array([])
times_b = 0
xr = -0.7; xl = -0.4 # initial guess
for i in range(0, 100):
    times_b += 1
    xc = (xr + xl)/2
    A2 = np.append(A2, xc)
    fc = xc*np.sin(3*xc) - np.exp(xc)
    if ( fc < 0 ):
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
A5 = 3*x - 4*y
A5 = A5.reshape(-1)
A6 = np.matmul(A,x).reshape(-1)
A7 = np.matmul(B,(x - y)).reshape(-1)
A8 = np.matmul(D,x).reshape(-1)
A9 = np.matmul(D,y) + z
A9 = A9.reshape(-1)
A10 = np.matmul(A,B)
A11 = np.matmul(B,C)
A12 = np.matmul(C,D)
