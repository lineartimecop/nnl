import random, math
import matplotlib.pyplot as plt

X = list(range(-50, 50))
Y = [0.05 * x ** 2 + 40 * (random.random() - 0.5) for x in X]


def mean(X):
    return sum(X) / len(X)


def sd(X):
    m = mean(X)

    return math.sqrt(1 / len(X) * sum([(x - m) ** 2 for x in X]))


def normalize(X):
    N = len(X)

    m = mean(X)
    standard_deviation = sd(X)

    return [(x - m) / standard_deviation for x in X]


def relu(x):
    return -0.25 * x if(x < 0) else x


def relu_prime(x):
    return -0.25 if(x < 0) else 1


def predict(w11, w12, w21, w22, b11, b12, b21, X, return_hidden = False):
    Z11 = [w11 * x + b11 for x in X]
    Z12 = [w12 * x + b12 for x in X]
    A11 = [relu(x) for x in Z11]
    A12 = [relu(x) for x in Z12]

    Y_hat = [w21 * x + w22 * y + b21 for x, y in zip(A11, A12)]

    if(return_hidden):
        return Z11, Z12, A11, A12, Y_hat
    else:
        return Y_hat


def train(X, Y, iterations = 10000, learning_rate = 0.0075, debug = False):
    print("\nNN trained in " + str(iterations) + " iterations with a learning rate of " + str(learning_rate) + "\n")

    printing = False
    gradient_check = False

    # Initialize parameters
    m = len(X)

    w11 = math.sqrt(2) * 2 * (random.random() - 0.5)
    w12 = math.sqrt(2) * 2 * (random.random() - 0.5)
    w21 = 2 * (random.random() - 0.5)
    w22 = 2 * (random.random() - 0.5)
    b11 = 0
    b12 = 0
    b21 = 0

    for i in range(iterations):
        if(printing):
            print("\nIteration " + str(i) + "\n")
            print(w11, w12, w21, w22, b11, b12, b21)

        # Forward propagation
        Z11, Z12, A11, A12, Y_hat = predict(w11, w12, w21, w22, b11, b12, b21, X, True)

        # Cost
        Y_hat_Y = [x - y for x, y in zip(Y_hat, Y)]
        J = 1 / m * sum([x * x for x in Y_hat_Y])

        if(printing):
            print("J(w, b): " + str(J))

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
            epsilon = 0.0000001

            Y_hat_plus = predict(w11 + epsilon, w12, w21, w22, b11, b12, b21, X)
            Y_hat_Y_plus = [x - y for x, y in zip(Y_hat_plus, Y)]
            Y_hat_minus = predict(w11 - epsilon, w12, w21, w22, b11, b12, b21, X)
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

    print("Train MSE: " + str(J))

    return w11, w12, w21, w22, b11, b12, b21


config = "Plot"

if(config == "Test"):
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    w11, w12, w21, w22, b11, b12, b21 = train(X_norm, Y_norm)
    Y_hat_norm = predict(w11, w12, w21, w22, b11, b12, b21, X_norm)

    # Costs
    epsilon = 0.01

    Y_hat_Y = [x - y for x, y in zip(Y_hat_norm, Y_norm)]
    J = 1 / len(X) * sum([x * x for x in Y_hat_Y])

    Y_hat_norm_plus = predict(w11, w12, w21, w22, b11, b12, b21 * (1 + epsilon), X_norm)

    Y_hat_Y_plus = [x - y for x, y in zip(Y_hat_norm_plus, Y_norm)]
    J_plus = 1 / len(X) * sum([x * x for x in Y_hat_Y_plus])

    Y_hat_norm_minus = predict(w11, w12, w21, w22, b11, b12, b21 * (1 - epsilon), X_norm)

    Y_hat_Y_minus = [x - y for x, y in zip(Y_hat_norm_minus, Y_norm)]
    J_minus = 1 / len(X) * sum([x * x for x in Y_hat_Y_minus])

    if((J <= J_plus) & (J <= J_minus)):
        print("OK: local minimum found.")
    else:
        print(J, J_plus, J_minus)
        print("ERROR: local minimum not found!")

if(config == "Debug"):
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    w11, w12, w21, w22, b11, b12, b21 = train(X_norm, Y_norm, iterations = 1, debug = True)

if(config == "Plot"):
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    w11, w12, w21, w22, b11, b12, b21 = train(X_norm, Y_norm)
    Y_hat_norm = predict(w11, w12, w21, w22, b11, b12, b21, X_norm)

    mean_Y = mean(Y)
    sd_Y = sd(Y)
    Y_hat = [x * sd_Y + mean_Y for x in Y_hat_norm]

    #print("X: " + str(X))
    #print("Y: " + str(Y))
    #print("Y_hat: " + str(Y_hat))

    plt.plot(X, Y, "o")
    plt.plot(X, Y_hat, "r-")
    plt.show()
