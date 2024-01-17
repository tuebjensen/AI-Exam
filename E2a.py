import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(a):
    return a*(1-a)

def forward(w1, w2, b1, b2, x):
    z1 = w1 @ x.reshape(-1, 1) + b1
    a1 = sigmoid(z1)
    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)
    return a1, a2

def backprop(w1, w2, b1, b2, x, y, h, a1, a2, alpha):
    error = a2 - y
    delta_a2 = error * dsigmoid(a2)
    delta_w2 = delta_a2 * a1.T

    delta_a1 = delta_a2 * w2 @ dsigmoid(a1)
    delta_w1 = np.dot(x.reshape(-1,1), delta_a1)

    w2 -= alpha * delta_w2
    w1 -= alpha * delta_w1
    b2 -= alpha * delta_a2
    b1 -= alpha * delta_a1
    return w1, w2, b1, b2

y = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float32)
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
w1 = np.array([[0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1]])
w2 = np.array([[0.1, 0.1, 0.1]])
b1 = np.array([0.1, 0.1, 0.1]).reshape(3, 1)
b2 = np.array([0.1]).reshape(1, 1)

max_epochs = 10000
mse_array = []
for i in range(max_epochs):
    for j in range(len(x)):
        a1, a2 = forward(w1, w2, b1, b2, x[j])
        w1, w2, b1, b2 = backprop(w1, w2, b1, b2, x[j], y[j], a2, a1, a2, alpha=0.8)
    running_se = 0
    for j in range(len(x)):
        a1, a2 = forward(w1, w2, b1, b2, x[j])
        # print(a2-y[j])
        running_se += np.square(a2 - y[j])
    mse_array.append(running_se[0,0]/len(x))

for i in range(len(x)):
    a1, a2 = forward(w1, w2, b1, b2, x[i])
    print(f'{x[i]}: {a2}')

plt.plot(mse_array)
plt.title("Mean Squared Error over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()