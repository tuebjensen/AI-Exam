import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))




def forward(w1, w2, x):
    z1 = w1 @ np.atleast_2d(x).T
    a1 = sigmoid(z1)
    z2 = w2 @ a1
    a2 = sigmoid(z2)
    return a1, a2

# def backprop(w1, w2, x, y, h, a1):
#     alpha = 0.1
#     delta = alpha * (h - y)# a scalar
#     w1 -= delta*w2*x # a vector times scalar
#     w2 -= delta*a1   # a vector times scalar
#     return w1, w2 # returns 2 vectors

def backprop(w1, w2, b1, b2, x, y, h, a1, a2, alpha):
    delta = alpha * (h - y)
    w2 -= delta * a2 * (1-a2) * a1.T
    w1 -= delta * a2 * (1-a2) * w2 * a1 * (1 - a1) * x

    return w1, w2

y = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float32)
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
w1 = np.array([[0.0, 0.1, 0.1],
               [0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1]])
w2 = np.array([[0.1, 0.1, 0.1]])

max_epochs = 1000

for i in range(max_epochs):
    for j in range(len(x)):
        a1, a2 = forward(w1, w2, x[j])
        # w1, w2 = backprop(w1, w2, x[j], y[j], a2, a1)
        w1, w2 = backprop(w1, w2, x[j], y[j], a2, a1, a2, alpha=0.03)

for i in range(len(x)):
    a1, a2 = forward(w1, w2, x[i])
    print(f'{x[i]}: {a2}')

