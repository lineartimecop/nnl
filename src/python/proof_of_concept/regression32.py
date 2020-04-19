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


def predict(X, w1_1, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1, details = False):
    Z1_1 = [w1_1 * x + b1_1 for x in X]
    Z1_2 = [w1_2 * x + b1_2 for x in X]
    Z1_3 = [w1_3 * x + b1_3 for x in X]
    A1_1 = [relu(x) for x in Z1_1]
    A1_2 = [relu(x) for x in Z1_2]
    A1_3 = [relu(x) for x in Z1_3]

    Z2_1 = [w2_11 * x + w2_12 * y + w2_13 * z + b2_1 for x, y, z in zip(A1_1, A1_2, A1_3)]
    Z2_2 = [w2_21 * x + w2_22 * y + w2_23 * z + b2_2 for x, y, z in zip(A1_1, A1_2, A1_3)]
    Z2_3 = [w2_31 * x + w2_32 * y + w2_33 * z + b2_3 for x, y, z in zip(A1_1, A1_2, A1_3)]
    A2_1 = [relu(x) for x in Z2_1]
    A2_2 = [relu(x) for x in Z2_2]
    A2_3 = [relu(x) for x in Z2_3]

    Y_hat = [w3_1 * x + w3_2 * y + w3_3 * z + b3_1 for x, y, z in zip(A2_1, A2_2, A2_3)]

    if(details):
        return Y_hat, Z1_1, Z1_2, Z1_3, Z2_1, Z2_2, Z2_3, A1_1, A1_2, A1_3, A2_1, A2_2, A2_3
    else:
        return Y_hat


def train(X, Y, epochs = 3000, learning_rate = 0.0075, gradient_check = False):
    # Initialize parameters
    m = len(X)

    w1_1 = 2 * (random.random() - 0.5)
    w1_2 = 2 * (random.random() - 0.5)
    w1_3 = 2 * (random.random() - 0.5)
    w2_11 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_12 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_13 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_21 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_22 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_23 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_31 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_32 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w2_33 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w3_1 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w3_2 = math.sqrt(3) * 2 * (random.random() - 0.5)
    w3_3 = math.sqrt(3) * 2 * (random.random() - 0.5)

    b1_1 = 0
    b1_2 = 0
    b1_3 = 0
    b2_1 = 0
    b2_2 = 0
    b2_3 = 0
    b3_1 = 0

    for i in range(epochs):
        # Forward propagation
        Y_hat, Z1_1, Z1_2, Z1_3, Z2_1, Z2_2, Z2_3, A1_1, A1_2, A1_3, A2_1, A2_2, A2_3 = predict(X, w1_1, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1, details = True)

        # Cost
        Y_hat_Y = [x - y for x, y in zip(Y_hat, Y)]
        J = 1 / m * sum([x * x for x in Y_hat_Y])

        # Back prop
        dA2_1 = [2 * x * w3_1 for x in Y_hat_Y]
        dA2_2 = [2 * x * w3_2 for x in Y_hat_Y]
        dA2_3 = [2 * x * w3_3 for x in Y_hat_Y]
        dw3_1 = 1 / m * sum([2 * x * y for x, y in zip(Y_hat_Y, A2_1)])
        dw3_2 = 1 / m * sum([2 * x * y for x, y in zip(Y_hat_Y, A2_2)])
        dw3_3 = 1 / m * sum([2 * x * y for x, y in zip(Y_hat_Y, A2_3)])
        db3_1 = 1 / m * sum([2 * x for x in Y_hat_Y])

        dZ2_1 = [x * relu_prime(y) for x, y in zip(dA2_1, Z2_1)]
        dZ2_2 = [x * relu_prime(y) for x, y in zip(dA2_2, Z2_2)]
        dZ2_3 = [x * relu_prime(y) for x, y in zip(dA2_3, Z2_3)]
        dA1_1 = [x * w2_11 + y * w2_21 + z * w2_31 for x, y, z in zip(dZ2_1, dZ2_2, dZ2_3)]
        dA1_2 = [x * w2_12 + y * w2_22 + z * w2_32 for x, y, z in zip(dZ2_1, dZ2_2, dZ2_3)]
        dA1_3 = [x * w2_13 + y * w2_23 + z * w2_33 for x, y, z in zip(dZ2_1, dZ2_2, dZ2_3)]
        dw2_11 = 1 / m * sum([x * y for x, y in zip(dZ2_1, A1_1)])
        dw2_12 = 1 / m * sum([x * y for x, y in zip(dZ2_1, A1_2)])
        dw2_13 = 1 / m * sum([x * y for x, y in zip(dZ2_1, A1_3)])
        dw2_21 = 1 / m * sum([x * y for x, y in zip(dZ2_2, A1_1)])
        dw2_22 = 1 / m * sum([x * y for x, y in zip(dZ2_2, A1_2)])
        dw2_23 = 1 / m * sum([x * y for x, y in zip(dZ2_2, A1_3)])
        dw2_31 = 1 / m * sum([x * y for x, y in zip(dZ2_3, A1_1)])
        dw2_32 = 1 / m * sum([x * y for x, y in zip(dZ2_3, A1_2)])
        dw2_33 = 1 / m * sum([x * y for x, y in zip(dZ2_3, A1_3)])
        db2_1 = 1 / m * sum(dZ2_1)
        db2_2 = 1 / m * sum(dZ2_2)
        db2_3 = 1 / m * sum(dZ2_3)

        dZ1_1 = [x * relu_prime(y) for x, y in zip(dA1_1, Z1_1)]
        dZ1_2 = [x * relu_prime(y) for x, y in zip(dA1_2, Z1_2)]
        dZ1_3 = [x * relu_prime(y) for x, y in zip(dA1_3, Z1_3)]
        dw1_1 = 1 / m * sum([x * y for x, y in zip(dZ1_1, X)])
        dw1_2 = 1 / m * sum([x * y for x, y in zip(dZ1_2, X)])
        dw1_3 = 1 / m * sum([x * y for x, y in zip(dZ1_3, X)])
        db1_1 = 1 / m * sum(dZ1_1)
        db1_2 = 1 / m * sum(dZ1_2)
        db1_3 = 1 / m * sum(dZ1_3)

        # Gradient check
        if(gradient_check):
            epsilon = 0.000001

            Y_hat_plus = predict(X, w1_1 + epsilon, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1)
            Y_hat_Y_plus = [x - y for x, y in zip(Y_hat_plus, Y)]
            Y_hat_minus = predict(X, w1_1 - epsilon, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1)
            Y_hat_Y_minus = [x - y for x, y in zip(Y_hat_minus, Y)]
            J_plus = 1 / m * sum([x * x for x in Y_hat_Y_plus])
            J_minus = 1 / m * sum([x * x for x in Y_hat_Y_minus])

            gradient_approximation = (J_plus - J_minus) / (2 * epsilon)
            print(gradient_approximation, dw1_1, gradient_approximation - dw1_1)

        # Update parameters
        w1_1 = w1_1 - learning_rate * dw1_1
        w1_2 = w1_2 - learning_rate * dw1_2
        w1_3 = w1_3 - learning_rate * dw1_3
        w2_11 = w2_11 - learning_rate * dw2_11
        w2_12 = w2_12 - learning_rate * dw2_12
        w2_13 = w2_13 - learning_rate * dw2_13
        w2_21 = w2_21 - learning_rate * dw2_21
        w2_22 = w2_22 - learning_rate * dw2_22
        w2_23 = w2_23 - learning_rate * dw2_23
        w2_31 = w2_31 - learning_rate * dw2_31
        w2_32 = w2_32 - learning_rate * dw2_32
        w2_33 = w2_33 - learning_rate * dw2_33
        w3_1 = w3_1 - learning_rate * dw3_1
        w3_2 = w3_2 - learning_rate * dw3_2
        w3_3 = w3_3 - learning_rate * dw3_3
        b1_1 = b1_1 - learning_rate * db1_1
        b1_2 = b1_2 - learning_rate * db1_2
        b1_3 = b1_3 - learning_rate * db1_3
        b2_1 = b2_1 - learning_rate * db2_1
        b2_2 = b2_2 - learning_rate * db2_2
        b2_3 = b2_3 - learning_rate * db2_3
        b3_1 = b3_1 - learning_rate * db3_1

    return w1_1, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1, J


X = list(range(-50, 50))
Y = [0.05 * x ** 2 + 40 * (random.random() - 0.5) for x in X]

X_norm = normalize(X)
Y_norm = normalize(Y)

w1_1, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1, J = train(X_norm, Y_norm)
Y_hat_norm = predict(X_norm, w1_1, w1_2, w1_3, w2_11, w2_12, w2_13, w2_21, w2_22, w2_23, w2_31, w2_32, w2_33, w3_1, w3_2, w3_3, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b3_1)

mean_Y = mean(Y)
sd_Y = sd(Y)
Y_hat = [x * sd_Y + mean_Y for x in Y_hat_norm]

print("J = " + str(J))

plt.plot(X, Y, "o")
plt.plot(X, Y_hat, "r-")
plt.show()
