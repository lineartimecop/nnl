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


def predict(w1, b1, X):
    Y_hat = [w1 * x + b1 for x in X]

    return Y_hat


def train(X, Y, epochs = 100, learning_rate = 0.0075, gradient_check = False):
    # Initialize parameters
    m = len(X)

    w1 = math.sqrt(2) * random.random()
    b1 = 0

    for i in range(epochs):
        # Forward propagation
        Y_hat = predict(w1, b1, X)

        # Cost
        Y_hat_Y = [x - y for x, y in zip(Y_hat, Y)]
        J = 1 / m * sum([x * x for x in Y_hat_Y])

        # Back prop
        dw1 = 1 / m * sum([2 * x * y for x, y in zip(Y_hat_Y, X)])
        db1 = 1 / m * sum([2 * x for x in Y_hat_Y])

        # Gradient check
        if(gradient_check):
            epsilon = 0.000001

            Y_hat_plus = predict(w1 + epsilon, b1, X)
            Y_hat_Y_plus = [x - y for x, y in zip(Y_hat_plus, Y)]
            Y_hat_minus = predict(w1 - epsilon, b1, X)
            Y_hat_Y_minus = [x - y for x, y in zip(Y_hat_minus, Y)]
            J_plus = 1 / m * sum([x * x for x in Y_hat_Y_plus])
            J_minus = 1 / m * sum([x * x for x in Y_hat_Y_minus])

            gradient_approximation = (J_plus - J_minus) / (2 * epsilon)
            print(gradient_approximation, dw1, gradient_approximation - dw1)

        # Update parameters
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1

    return w1, b1, J


X = list(range(1, 10))
Y = [0.6 * x + 2 + 2 * (random.random() - 0.5) for x in X]

X_norm = normalize(X)
Y_norm = normalize(Y)

w1, b1, J = train(X_norm, Y_norm)
Y_hat_norm = predict(w1, b1, X_norm)

mean_Y = mean(Y)
sd_Y = sd(Y)
Y_hat = [x * sd_Y + mean_Y for x in Y_hat_norm]

print("J = " + str(J))

plt.plot(X, Y, "o")
plt.plot(X, Y_hat, "r-")
plt.show()
