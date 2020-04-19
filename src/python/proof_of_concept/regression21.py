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


def relu(x):
    return -0.25 * x if(x < 0) else x


def relu_prime(x):
    return -0.25 if(x < 0) else 1


def predict(X, w11, w12, w21, w22, b11, b12, b21, details = False):
    Z11 = [w11 * x + b11 for x in X]
    Z12 = [w12 * x + b12 for x in X]
    A11 = [relu(x) for x in Z11]
    A12 = [relu(x) for x in Z12]

    Y_hat = [w21 * x + w22 * y + b21 for x, y in zip(A11, A12)]

    if(details):
        return Y_hat, Z11, Z12, A11, A12
    else:
        return Y_hat


def train(X, Y, epochs = 3000, learning_rate = 0.0075, gradient_check = False):
    # Initialize parameters
    m = len(X)

    w11 = math.sqrt(2) * 2 * (random.random() - 0.5)
    w12 = math.sqrt(2) * 2 * (random.random() - 0.5)
    w21 = 2 * (random.random() - 0.5)
    w22 = 2 * (random.random() - 0.5)
    b11 = 0
    b12 = 0
    b21 = 0

    for i in range(epochs):
        # Forward propagation
        Y_hat, Z11, Z12, A11, A12 = predict(X, w11, w12, w21, w22, b11, b12, b21, details = True)

        # Cost
        Y_hat_Y = [x - y for x, y in zip(Y_hat, Y)]
        J = 1 / m * sum([x * x for x in Y_hat_Y])

        # Back prop
        dA11 = [2 * x * w21 for x in Y_hat_Y]
        dA12 = [2 * x * w22 for x in Y_hat_Y]
        dw21 = 1 / m * sum([2 * x * y for x, y in zip(Y_hat_Y, A11)])
        dw22 = 1 / m * sum([2 * x * y for x, y in zip(Y_hat_Y, A12)])
        db21 = 1 / m * sum([2 * x for x in Y_hat_Y])

        dZ11 = [x * relu_prime(y) for x, y in zip(dA11, Z11)]
        dZ12 = [x * relu_prime(y) for x, y in zip(dA12, Z12)]
        dw11 = 1 / m * sum([x * y for x, y in zip(dZ11, X)])
        dw12 = 1 / m * sum([x * y for x, y in zip(dZ12, X)])
        db11 = 1 / m * sum(dZ11)
        db12 = 1 / m * sum(dZ12)

        # Gradient check
        if(gradient_check):
            epsilon = 0.000001

            Y_hat_plus = predict(X, w11 + epsilon, w12, w21, w22, b11, b12, b21)
            Y_hat_Y_plus = [x - y for x, y in zip(Y_hat_plus, Y)]
            Y_hat_minus = predict(X, w11 - epsilon, w12, w21, w22, b11, b12, b21)
            Y_hat_Y_minus = [x - y for x, y in zip(Y_hat_minus, Y)]
            J_plus = 1 / m * sum([x * x for x in Y_hat_Y_plus])
            J_minus = 1 / m * sum([x * x for x in Y_hat_Y_minus])

            gradient_approximation = (J_plus - J_minus) / (2 * epsilon)
            print(gradient_approximation, dw11, gradient_approximation - dw11)

        # Update parameters
        w11 = w11 - learning_rate * dw11
        w12 = w12 - learning_rate * dw12
        w21 = w21 - learning_rate * dw21
        w22 = w22 - learning_rate * dw22
        b11 = b11 - learning_rate * db11
        b12 = b12 - learning_rate * db12
        b21 = b21 - learning_rate * db21

    return w11, w12, w21, w22, b11, b12, b21, J


X = list(range(-50, 50))
Y = [0.05 * x ** 2 + 40 * (random.random() - 0.5) for x in X]

X_norm = normalize(X)
Y_norm = normalize(Y)

w11, w12, w21, w22, b11, b12, b21, J = train(X_norm, Y_norm)
Y_hat_norm = predict(X_norm, w11, w12, w21, w22, b11, b12, b21)

mean_Y = mean(Y)
sd_Y = sd(Y)
Y_hat = [x * sd_Y + mean_Y for x in Y_hat_norm]

print("J = " + str(J))

plt.plot(X, Y, "o")
plt.plot(X, Y_hat, "r-")
plt.show()
