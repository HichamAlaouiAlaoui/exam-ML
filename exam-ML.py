import numpy as np
import matplotlib as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_inference (x,w,b):
    z = x @ w + b
    p=sigmoid(z)
    return p

def logreg_train(X, Y):
    # X[0, :]  -> Y[0]
    # X[1, :]  -> Y[1]
    # ...
    m=X.shape[0]
    n=X.shape[1]
    w=np.zeros(n)
    b=0
    for step in range (1000):
        P= logreg_inference(X, w, b)
        grad_b = (P - Y).mean()
        grad_W = (X.T @ (P - Y)) / m
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

data = np.loadtxt("exam.txt")
X=data[:, :2]
Y= data[:, 2]
w, b = logreg_train(X, Y)
print(w)
print(b)

#x=np.array([50,25], [40,12], [60,40])
#w=np.array([0.1, -1])
#b=-0.5
#p= logreg_inference(x,w,b)
print(p)
