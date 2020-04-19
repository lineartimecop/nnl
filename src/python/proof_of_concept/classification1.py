import random, math
import matplotlib.pyplot as plt

def mean(X):
    return sum(X) / len(X)


def sd(X):
    m = mean(X)

    return math.sqrt(1 / len(X) * sum([(x - m) ** 2 for x in X]))


def normalize(X):
    m = mean(X)
    standard_deviation = sd(X)

    return [(x - m) / standard_deviation for x in X]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    return math.exp(-x) / ((1 + math.exp(-x)) ** 2)


def predict(X, w1, b1, details = False):
    Z1 = [w1 * x + b1 for x in X]
    Y_hat = [sigmoid(z1) for z1 in Z1]

    if(details):
        return Y_hat, Z1
    else:
        return Y_hat


def train(X, Y, epochs = 100, learning_rate = 10, gradient_check = False):
    # Initialize parameters
    m = len(X)

    w1 = math.sqrt(2) * random.random()
    b1 = 0

    for i in range(epochs):
        # Forward propagation
        Y_hat, Z1 = predict(X, w1, b1, details = True)

        # Cost
        L = [-(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat)) for y, y_hat in zip(Y, Y_hat)]
        J = 1 / m * sum(L)

        # Back prop
        dY_hat = [-(y / y_hat - (1 - y) / (1 - y_hat)) for y, y_hat in zip(Y, Y_hat)]
        dZ1 = [dy_hat * sigmoid_prime(z1) for dy_hat, z1 in zip(dY_hat, Z1)]

        dw1 = 1 / m * sum([dz1 * x for dz1, x in zip(dZ1, X)])
        db1 = 1 / m * sum(dZ1)

        # Gradient check
        if(gradient_check):
            epsilon = 0.000001

            Y_hat_plus = predict(X, w1 + epsilon, b1)
            L_plus = [-(y * math.log(y_hat_plus) + (1 - y) * math.log(1 - y_hat_plus)) for y, y_hat_plus in zip(Y, Y_hat_plus)]
            Y_hat_minus = predict(X, w1 - epsilon, b1)
            L_minus = [-(y * math.log(y_hat_minus) + (1 - y) * math.log(1 - y_hat_minus)) for y, y_hat_minus in zip(Y, Y_hat_minus)]
            J_plus = 1 / m * sum(L_plus)
            J_minus = 1 / m * sum(L_minus)

            gradient_approximation = (J_plus - J_minus) / (2 * epsilon)
            print(gradient_approximation, dw1, gradient_approximation - dw1)

        # Update parameters
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1

    return w1, b1, J


X = list(range(0, 10))
Y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

X_norm = normalize(X)

w1, b1, J = train(X_norm, Y)
Y_hat = predict(X_norm, w1, b1)

print("J = " + str(J))

plt.plot(X, Y, "o")
plt.plot(X, Y_hat, "r-")
plt.show()
