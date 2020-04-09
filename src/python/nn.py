import numpy as np
import matplotlib.pyplot as plt
import datetime, os, shutil

m_linear = 10
X_linear = np.arange(-m_linear / 2, m_linear / 2).reshape(1, m_linear)
Y_linear = 5 * (X_linear - 10) + 60# + 10 * (np.random.rand(1, m_linear) - 0.5)

m_non_linear = 100
X_non_linear = np.arange(-m_non_linear / 2, m_non_linear / 2).reshape(1, m_non_linear)
Y_non_linear = 0.0004 * (X_non_linear - 25) ** 3 + 0.02 * (X_non_linear - 30) ** 2# + 10 * (np.random.rand(1, m_non_linear) - 0.5)

m_very_wiggly = 1000
X_very_wiggly = np.arange(-m_very_wiggly / 2, m_very_wiggly / 2).reshape(1, m_very_wiggly)
Y_very_wiggly = 0.0002 * X_very_wiggly ** 2 + 10 * np.sin(X_very_wiggly / 25) + 10 * np.sin(X_very_wiggly / 50) + 10 * np.sin(X_very_wiggly / 100) + 10 * np.sin(X_very_wiggly / 200) + 10 * np.sin(X_very_wiggly / 400)# + 10 * (np.random.rand(1, m_very_wiggly) - 0.5)

X = X_very_wiggly
Y = Y_very_wiggly


def normalize(X):
    N = X.shape[1]

    m = np.mean(X)
    standard_deviation = np.std(X)

    return (X - m) / standard_deviation


def relu(x):
    return -0.1 * x if(x < 0) else x


relu = np.vectorize(relu)


def relu_prime(x):
    return -0.1 if(x < 0) else 1.0


relu_prime = np.vectorize(relu_prime)


def predict(W, b, X, return_hidden = False):
    Z = []
    A = []
    L = len(W) - 1

    if(L < 1):
        raise Exception("NN must contain at least one hidden layer in addition to the output layer!")

    if(len(W) != len(b)):
        raise Exception("Length of W and b must be equal!")

    # Input layer
    Z.append(np.dot(W[0], X) + b[0])
    A.append(relu(Z[0]))

    # l-th hidden layer
    for l in range(1, L):
        Z.append(np.dot(W[l], A[l - 1]) + b[l])
        A.append(relu(Z[l]))

    # Output layer
    Y_hat = np.dot(W[L], A[L - 1]) + b[L]

    if(return_hidden):
        return Z, A, Y_hat
    else:
        return Y_hat


def train(X, Y, depth, width, iterations = 3000, learning_rate = None, alpha = 0.9, mini_batch_size = None, grad_check = False, debug = False, printing = False):
    # Initialize parameters
    W = []
    b = []
    L = depth
    m = X.shape[1]
    n = X.shape[0]

    if(debug):
        training_data = []

    # - Input layer
    W.append(1 / np.sqrt(n) * 2 * (np.random.rand(width, n) - 0.5))
    b.append(np.zeros((width, n)))

    # - l-th hidden layer
    for l in range(1, L):
        W.append(1 / np.sqrt(width) * 2 * (np.random.rand(width, width) - 0.5))
        b.append(np.zeros((width, 1)))

    # - Output layer
    W.append(1 / np.sqrt(width) * 2 * (np.random.rand(1, width) - 0.5))
    b.append(np.zeros((1, 1)))

    # Length of gradient vector
    p_num = 0
    for l in range(L + 1):
        p_num = p_num + W[l].shape[0] * W[l].shape[1] + b[l].shape[0] * b[l].shape[1]

    # Shuffle data for mini-batches
    if(mini_batch_size):
        D = np.concatenate((X, Y), axis = 0)
        np.random.shuffle(D.T)

        X_shuffled = D[0:X.shape[0], 0:X.shape[1]]
        Y_shuffled = D[X.shape[0]:X.shape[0] + Y.shape[0], 0:Y.shape[1]]

        mini_batch_index = 0

    v_W = []
    v_b = []

    # Training NN
    for i in range(iterations):
        # Take one mini-batch
        if(mini_batch_size):
            X_mini = X_shuffled[:,mini_batch_index * mini_batch_size:(mini_batch_index + 1) * mini_batch_size]
            Y_mini = Y_shuffled[:,mini_batch_index * mini_batch_size:(mini_batch_index + 1) * mini_batch_size]

            if((mini_batch_index + 1) * mini_batch_size >= m):
                mini_batch_index = 0
            else:
                mini_batch_index = mini_batch_index + 1
        else:
            X_mini = X
            Y_mini = Y

        # Forward propagation
        Z, A, Y_hat_mini = predict(W, b, X_mini, True)

        # Cost
        J = 1 / m * np.dot(Y_hat_mini - Y_mini, (Y_hat_mini - Y_mini).T)

        # Back prop
        dA = []
        dZ = []
        dW = []
        db = []

        # - Output layer
        dA.append(np.dot(W[L].T, 2 * (Y_hat_mini - Y_mini)))
        dW.append(2 / m * np.dot(Y_hat_mini - Y_mini, A[L - 1].T))
        db.append(2 / m * np.sum(Y_hat_mini - Y_mini, axis = 1, keepdims = True))

        # - l-th hidden layer
        for l in range(1, L):
            dZ.append(np.multiply(dA[l - 1], relu_prime(Z[L - l])))
            dA.append(np.dot(W[L - l].T, dZ[l - 1]))
            dW.append(1 / m * np.dot(dZ[l - 1], A[L - l - 1].T))
            db.append(1 / m * np.sum(dZ[l - 1], axis = 1, keepdims = True))

        # - Input layer
        dZ.append(np.multiply(dA[L - 1], relu_prime(Z[0])))
        dW.append(1 / m * np.sum(np.multiply(np.repeat(X_mini, width, axis = 0), dZ[L - 1]), axis = 1, keepdims = True))
        db.append(1 / m * np.sum(dZ[L - 1], axis = 1, keepdims = True))

        # Grad check
        if(grad_check):
            grad = np.zeros((1, p_num))
            grad_approx = np.zeros((1, p_num))
            eps_grad_check = 0.0000001

            # - create grad vector
            index = 0
            for l in range(L + 1):
                for j in range(len(W[l])):
                    for k in range(len(W[l][j])):
                        grad[0][index] = dW[L - l][j][k]
                        index = index + 1

                for j in range(len(b[l])):
                    grad[0][index] = db[L - l][j]
                    index = index + 1

            # - calculate grad approx
            index = 0
            for l in range(L + 1):
                W_plus = []
                W_minus = []

                b_plus = []
                b_minus = []

                for j in range(L + 1):
                    W_plus.append(np.copy(W[j]))
                    W_minus.append(np.copy(W[j]))

                    b_plus.append(np.copy(b[j]))
                    b_minus.append(np.copy(b[j]))

                # dW
                for j in range(len(W[l])):
                    for k in range(len(W[l][j])):
                        W_plus[l][j][k] = W_plus[l][j][k] + eps_grad_check
                        W_minus[l][j][k] = W_minus[l][j][k] - eps_grad_check

                        Y_hat_plus = predict(W_plus, b, X_mini)
                        Y_hat_minus = predict(W_minus, b, X_mini)
                        J_plus = 1 / m * np.dot(Y_hat_plus - Y_mini, (Y_hat_plus - Y_mini).T)
                        J_minus = 1 / m * np.dot(Y_hat_minus - Y_mini, (Y_hat_minus - Y_mini).T)

                        W_plus[l][j][k] = W_plus[l][j][k] - eps_grad_check
                        W_minus[l][j][k] = W_minus[l][j][k] + eps_grad_check

                        grad_approx[0][index] = (J_plus - J_minus) / (2 * eps_grad_check)
                        index = index + 1

                # db
                for j in range(len(b[l])):
                    b_plus[l][j] = b_plus[l][j] + eps_grad_check
                    b_minus[l][j] = b_minus[l][j] - eps_grad_check

                    Y_hat_plus = predict(W, b_plus, X_mini)
                    Y_hat_minus = predict(W, b_minus, X_mini)
                    J_plus = 1 / m * np.dot(Y_hat_plus - Y_mini, (Y_hat_plus - Y_mini).T)
                    J_minus = 1 / m * np.dot(Y_hat_minus - Y_mini, (Y_hat_minus - Y_mini).T)

                    b_plus[l][j] = b_plus[l][j] - eps_grad_check
                    b_minus[l][j] = b_minus[l][j] + eps_grad_check

                    grad_approx[0][index] = (J_plus - J_minus) / (2 * eps_grad_check)
                    index = index + 1

            nom = np.linalg.norm(grad - grad_approx, 2)
            den = np.linalg.norm(grad, 2) + np.linalg.norm(grad_approx, 2)

            if(nom / den > eps_grad_check):
                print("Error with gradient in iteration = " + str(i) + ", grad check = " + str(nom / den))

        # Learning rate
        eps_momentum = learning_rate

        # Momentum
        if(len(v_W) == 0):
            for l in range(L + 1):
                v_W.append(-eps_momentum * dW[L - l])
        else:
            for l in range(L + 1):
                v_W[l] = alpha * v_W[l] - eps_momentum * dW[L - l]

        if(len(v_b) == 0):
            for l in range(L + 1):
                v_b.append(-eps_momentum * db[L - l])
        else:
            for l in range(L + 1):
                v_b[l] = alpha * v_b[l] - eps_momentum * db[L - l]

        # Collect training data
        if(debug):
            if(i == 0):
                header = []
                header.append("Iteration")
                header.append("Cost")
                header.append("Epsilon")

            row = []
            row.append(i + 1)
            row.append(J[0][0])
            row.append(eps_momentum)

            for l in range(L + 1):
                for p in range(W[l].shape[0]):
                    for q in range(W[l].shape[1]):
                        if(i == 0):
                            header.append("W_" + str(l + 1) + "_" + str(p + 1) + str(q + 1))
                            header.append("dW_" + str(l + 1) + "_" + str(p + 1) + str(q + 1))

                        row.append(W[l][p][q])
                        row.append(dW[L - l][p][q])

                for p in range(b[l].shape[0]):
                    if(i == 0):
                        header.append("b_" + str(l + 1) + "_" + str(p + 1))
                        header.append("db_" + str(l + 1) + "_" + str(p + 1))

                    row.append(b[l][p][0])
                    row.append(db[L - l][p][0])

            if(i == 0):
                training_data.append(header)

            training_data.append(row)

        # Update parameters
        for l in range(0, L + 1):
            W[l] = W[l] + v_W[l]
            b[l] = b[l] + v_b[l]

        # Printing
        if(printing and (i + 1) % 1000 == 0):
            print("Normalized MSE in iteration " + str(i + 1) + " = " + str(J[0][0]))

    # Dump training data
    if(debug):
        if os.path.exists("nn-debug"):
            shutil.rmtree("nn-debug")

        os.makedirs("nn-debug")

        # Text file
        with open("nn-debug/nn.txt", "w") as file:
            for i in range(len(training_data)):
                for j in range(len(training_data[0])):
                    if(j != 0):
                        file.write("\t")

                    file.write(str(training_data[i][j]))

                file.write("\n")

        # Plots
        print("Plotting parameters...")
        I = list(range(1, iterations + 1))
        for j in range(1, 3):
            P = []
            dP = []

            for i in range(1, len(training_data)):
                P.append(training_data[i][j])

            parameter_name = training_data[0][j]

            plt.plot(I, P, "r-")
            plt.xlabel("Iteration")
            plt.ylabel(parameter_name)
            plt.title(parameter_name + " by iteration")

            plt.savefig("nn-debug/" + parameter_name  + ".png")
            plt.close()

        for j in range(3, len(training_data[0]), 2):
            P = []
            dP = []

            for i in range(1, len(training_data)):
                P.append(training_data[i][j])
                dP.append(training_data[i][j + 1])

            parameter_name = training_data[0][j]
            d_parameter_name = training_data[0][j + 1]

            plt.subplot(211)
            plt.plot(I, P, "r-")
            plt.ylabel(parameter_name)
            plt.title(parameter_name + " by iteration")

            plt.subplot(212)
            plt.plot(I, dP, "b-")
            plt.xlabel("Iteration")
            plt.ylabel(d_parameter_name)

            plt.savefig("nn-debug/" + parameter_name  + ".png")
            plt.close()

    return W, b, J[0][0]


config = "Plot"

if(config == "MeasureParameter"):
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    params = []
    mse = []
    for i in range(100):
        N = 100
        average_mse = 0
        for j in range(N):
            p = 0.01 * (i + 1)
            W, b, tmp = train(X_norm, Y_norm, depth = 3, width = 5, iterations = 3000, param = p, mini_batch_size = 16)
            average_mse = average_mse + tmp

        average_mse = average_mse / N
        if(average_mse > 3):
            average_mse = 3

        print(str(datetime.datetime.now()) + ": MSE for base = " + str(p) + " is " + str(average_mse))

        params.append(p)
        mse.append(average_mse)

    plt.plot(params, mse, "r-")
    plt.xlabel("Base")
    plt.ylabel("Training MSE")
    plt.title("Training MSE by base")
    plt.show()

if(config == "Plot"):
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    # Linear function
    #W, b, train_mse = train(X_norm, Y_norm, depth = 1, width = 2, learning_rate = 0.12, iterations = 100)
    #W, b, train_mse = train(X_norm, Y_norm, depth = 1, width = 2, learning_rate = 0.12, iterations = 100, mini_batch_size = 2)

    # Non-linear function
    #W, b, train_mse = train(X_norm, Y_norm, depth = 3, width = 5, learning_rate = 0.025, iterations = 3000)
    #W, b, train_mse = train(X_norm, Y_norm, depth = 3, width = 5, learning_rate = 0.1, iterations = 3000, mini_batch_size = 16)

    # Very wiggly function
    #W, b, train_mse = train(X_norm, Y_norm, depth = 5, width = 10, learning_rate = 0.01, iterations = 30000, printing = True)
    W, b, train_mse = train(X_norm, Y_norm, depth = 5, width = 10, learning_rate = 0.03, iterations = 30000, mini_batch_size = 128, printing = True)

    Y_hat_norm = predict(W, b, X_norm)

    print("Train MSE: " + str(train_mse))

    mean_Y = np.mean(Y)
    sd_Y = np.std(Y)
    Y_hat = sd_Y * Y_hat_norm + mean_Y

    plt.plot(X[0], Y[0], "o")
    plt.plot(X[0], Y_hat[0], "r-")
    plt.show()
