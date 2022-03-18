import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))


def logreg_inference(x, w, b):
    """Inference step for the logistic regression model."""
    logit  = (x @ w) + b
    p = sigmoid(logit)
    return p


# x = np.array([1, 0, -2])
# w = np.array([0.5, 0.5, -1])
# b = -0.5
# p = logreg_inference(x, w, b)
# print(p)


def cross_entropy(P, Y):
    """Binary cross-entropy."""
    return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()


def logreg_train(X, Y):
    """Training procedure for the logistic regression model."""
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    lr = 0.01
    for step in range(100000):
        P = logreg_inference(X, w, b)
        if step % 1000 == 0:
            loss = cross_entropy(P, Y)
            print(step, loss)
        grad_w = (X.T @ (P - Y)) / m
        grad_b = (P - Y).mean()
        # Gradient descent updates.
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


# Load the data, train the mnodel and measure the accuracy.
data = np.loadtxt("exam.txt")
X = data[:, :-1]
Y = data[:, -1]
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
w, b = logreg_train(X, Y)
P = logreg_inference(X, w, b)
Yhat = (P >= 0.5)
accuracy = (Y == Yhat).mean()
print("Accuracy:", accuracy * 100)
