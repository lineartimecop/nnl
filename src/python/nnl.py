import numpy as np
import datetime, os, shutil


def linear_data(size = 10, noisiness = 0):
    if(size < 0):
        raise Exception("Size must be greater than 0!")

    if(noisiness < 0 or noisiness > 3):
        raise Exception("Noiseness must be 0, 1 or 2!")

    noise = np.zeros((1, size))

    if(noisiness == 1):
        noise = 5 * (np.random.rand(1, size) - 0.5)
    if(noisiness == 2):
        noise = 20 * (np.random.rand(1, size) - 0.5)
    if(noisiness == 3):
        noise = 50 * (np.random.rand(1, size) - 0.5)

    X = np.arange(-size / 2, size / 2).reshape(1, size)
    Y = 5 * (X - 10) + 60 + noise

    return X, Y


def non_linear_data(size = 100, noisiness = 0):
    if(size < 0):
        raise Exception("Size must be greater than 0!")

    if(noisiness < 0 or noisiness > 3):
        raise Exception("Noiseness must be 0, 1 or 2!")

    noise = np.zeros((1, size))

    if(noisiness == 1):
        noise = 5 * (np.random.rand(1, size) - 0.5)
    if(noisiness == 2):
        noise = 20 * (np.random.rand(1, size) - 0.5)
    if(noisiness == 3):
        noise = 50 * (np.random.rand(1, size) - 0.5)

    X = np.arange(-size / 2, size / 2).reshape(1, size)
    Y = 0.0004 * (X - 25) ** 3 + 0.02 * (X - 30) ** 2 + noise

    return X, Y


def very_wiggly_data(size = 1000, noisiness = 0):
    if(size < 0):
        raise Exception("Size must be greater than 0!")

    if(noisiness < 0 or noisiness > 3):
        raise Exception("Noiseness must be 0, 1 or 2!")

    noise = np.zeros((1, size))

    if(noisiness == 1):
        noise = 5 * (np.random.rand(1, size) - 0.5)
    if(noisiness == 2):
        noise = 20 * (np.random.rand(1, size) - 0.5)
    if(noisiness == 3):
        noise = 50 * (np.random.rand(1, size) - 0.5)

    X = np.arange(-size / 2, size / 2).reshape(1, size)
    Y = 0.0002 * X ** 2 + 10 * np.sin(X / 25) + 10 * np.sin(X / 50) + 10 * np.sin(X / 100) + 10 * np.sin(X / 200) + 10 * np.sin(X / 400) + noise

    return X, Y


def normalize(X):
    m = np.mean(X)
    standard_deviation = np.std(X)

    return (X - m) / standard_deviation


def relu(x):
    return -0.1 * x if(x < 0) else x


relu = np.vectorize(relu)


def relu_prime(x):
    return -0.1 if(x < 0) else 1.0


relu_prime = np.vectorize(relu_prime)


class NN:
    def __init__(self, depth, width):
        self.W = []
        self.b = []

        self.depth = depth
        self.width = width


    def train(self, X, Y, epochs = 3000, learning_rate = None, alpha = 0.9, batch_size = None, grad_check = False, debug = False, printing = False):
        # Initialize parameters
        L = self.depth
        m = X.shape[1]
        n = X.shape[0]

        if(debug):
            training_data = []

        # - Input layer
        self.W.append(1 / np.sqrt(n) * 2 * (np.random.rand(self.width, n) - 0.5))
        self.b.append(np.zeros((self.width, n)))

        # - l-th hidden layer
        for l in range(1, L):
            self.W.append(1 / np.sqrt(self.width) * 2 * (np.random.rand(self.width, self.width) - 0.5))
            self.b.append(np.zeros((self.width, 1)))

        # - Output layer
        self.W.append(1 / np.sqrt(self.width) * 2 * (np.random.rand(1, self.width) - 0.5))
        self.b.append(np.zeros((1, 1)))

        # Length of gradient vector
        p_num = 0
        for l in range(L + 1):
            p_num = p_num + self.W[l].shape[0] * self.W[l].shape[1] + self.b[l].shape[0] * self.b[l].shape[1]

        # Shuffle data for batches
        if(batch_size):
            D = np.concatenate((X, Y), axis = 0)
            np.random.shuffle(D.T)

            X_shuffled = D[0:X.shape[0], 0:X.shape[1]]
            Y_shuffled = D[X.shape[0]:X.shape[0] + Y.shape[0], 0:Y.shape[1]]

            batch_index = 0

        v_W = []
        v_b = []

        # Training NN
        for i in range(epochs):
            # Take one batch
            if(batch_size):
                X_batch = X_shuffled[:,batch_index * batch_size:(batch_index + 1) * batch_size]
                Y_batch = Y_shuffled[:,batch_index * batch_size:(batch_index + 1) * batch_size]

                if((batch_index + 1) * batch_size >= m):
                    batch_index = 0
                else:
                    batch_index = batch_index + 1
            else:
                X_batch = X
                Y_batch = Y

            # Forward propagation
            Z, A, Y_hat_batch = self.predict(X_batch, return_hidden = True)

            # Cost
            J = 1 / m * np.dot(Y_hat_batch - Y_batch, (Y_hat_batch - Y_batch).T)

            # Back propagation
            dA = []
            dZ = []
            dW = []
            db = []

            # - Output layer
            dA.append(np.dot(self.W[L].T, 2 * (Y_hat_batch - Y_batch)))
            dW.append(2 / m * np.dot(Y_hat_batch - Y_batch, A[L - 1].T))
            db.append(2 / m * np.sum(Y_hat_batch - Y_batch, axis = 1, keepdims = True))

            # - l-th hidden layer
            for l in range(1, L):
                dZ.append(np.multiply(dA[l - 1], relu_prime(Z[L - l])))
                dA.append(np.dot(self.W[L - l].T, dZ[l - 1]))
                dW.append(1 / m * np.dot(dZ[l - 1], A[L - l - 1].T))
                db.append(1 / m * np.sum(dZ[l - 1], axis = 1, keepdims = True))

            # - Input layer
            dZ.append(np.multiply(dA[L - 1], relu_prime(Z[0])))
            dW.append(1 / m * np.sum(np.multiply(np.repeat(X_batch, self.width, axis = 0), dZ[L - 1]), axis = 1, keepdims = True))
            db.append(1 / m * np.sum(dZ[L - 1], axis = 1, keepdims = True))

            # Gradient checking
            if(grad_check):
                grad = np.zeros((1, p_num))
                grad_approx = np.zeros((1, p_num))
                eps_grad_check = 0.0000001

                # - create grad vector
                index = 0
                for l in range(L + 1):
                    for j in range(len(self.W[l])):
                        for k in range(len(self.W[l][j])):
                            grad[0][index] = dW[L - l][j][k]
                            index = index + 1

                    for j in range(len(self.b[l])):
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
                        W_plus.append(np.copy(self.W[j]))
                        W_minus.append(np.copy(self.W[j]))

                        b_plus.append(np.copy(self.b[j]))
                        b_minus.append(np.copy(self.b[j]))

                    # dW
                    for j in range(len(self.W[l])):
                        for k in range(len(self.W[l][j])):
                            W_plus[l][j][k] = W_plus[l][j][k] + eps_grad_check
                            W_minus[l][j][k] = W_minus[l][j][k] - eps_grad_check

                            Y_hat_plus = self.predict(X_batch, W = W_plus)
                            Y_hat_minus = self.predict(X_batch, W = W_minus)
                            J_plus = 1 / m * np.dot(Y_hat_plus - Y_batch, (Y_hat_plus - Y_batch).T)
                            J_minus = 1 / m * np.dot(Y_hat_minus - Y_batch, (Y_hat_minus - Y_batch).T)

                            W_plus[l][j][k] = W_plus[l][j][k] - eps_grad_check
                            W_minus[l][j][k] = W_minus[l][j][k] + eps_grad_check

                            grad_approx[0][index] = (J_plus - J_minus) / (2 * eps_grad_check)
                            index = index + 1

                    # db
                    for j in range(len(self.b[l])):
                        b_plus[l][j] = b_plus[l][j] + eps_grad_check
                        b_minus[l][j] = b_minus[l][j] - eps_grad_check

                        Y_hat_plus = self.predict(X_batch, b = b_plus)
                        Y_hat_minus = self.predict(X_batch, b = b_minus)
                        J_plus = 1 / m * np.dot(Y_hat_plus - Y_batch, (Y_hat_plus - Y_batch).T)
                        J_minus = 1 / m * np.dot(Y_hat_minus - Y_batch, (Y_hat_minus - Y_batch).T)

                        b_plus[l][j] = b_plus[l][j] - eps_grad_check
                        b_minus[l][j] = b_minus[l][j] + eps_grad_check

                        grad_approx[0][index] = (J_plus - J_minus) / (2 * eps_grad_check)
                        index = index + 1

                nom = np.linalg.norm(grad - grad_approx, 2)
                den = np.linalg.norm(grad, 2) + np.linalg.norm(grad_approx, 2)

                if(nom / den > eps_grad_check):
                    print("Error with gradient in epoch " + str(i) + ": grad_check = " + str(nom / den))

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
                    header.append("Epoch")
                    header.append("Cost")
                    header.append("Epsilon")

                row = []
                row.append(i + 1)
                row.append(J[0][0])
                row.append(eps_momentum)

                for l in range(L + 1):
                    for p in range(self.W[l].shape[0]):
                        for q in range(self.W[l].shape[1]):
                            if(i == 0):
                                header.append("W_" + str(l + 1) + "_" + str(p + 1) + str(q + 1))
                                header.append("dW_" + str(l + 1) + "_" + str(p + 1) + str(q + 1))

                            row.append(self.W[l][p][q])
                            row.append(dW[L - l][p][q])

                    for p in range(self.b[l].shape[0]):
                        if(i == 0):
                            header.append("b_" + str(l + 1) + "_" + str(p + 1))
                            header.append("db_" + str(l + 1) + "_" + str(p + 1))

                        row.append(self.b[l][p][0])
                        row.append(db[L - l][p][0])

                if(i == 0):
                    training_data.append(header)

                training_data.append(row)

            # Update parameters
            for l in range(0, L + 1):
                self.W[l] = self.W[l] + v_W[l]
                self.b[l] = self.b[l] + v_b[l]

            # Printing
            if(printing and (i + 1) % 1000 == 0):
                print(str(datetime.datetime.now()) + ": " + str(J[0][0]) + " MSE in epoch " + str(i + 1))

        # Dump training data
        if(debug):
            if os.path.exists("debug"):
                shutil.rmtree("debug")

            os.makedirs("debug")

            # Text file
            with open("debug/nn.txt", "w") as file:
                for i in range(len(training_data)):
                    for j in range(len(training_data[0])):
                        if(j != 0):
                            file.write("\t")

                        file.write(str(training_data[i][j]))

                    file.write("\n")

            # Plots
            print("Plotting parameters...")
            I = list(range(1, epochs + 1))
            for j in range(1, 3):
                P = []
                dP = []

                for i in range(1, len(training_data)):
                    P.append(training_data[i][j])

                parameter_name = training_data[0][j]

                plt.plot(I, P, "r-")
                plt.xlabel("Epoch")
                plt.ylabel(parameter_name)
                plt.title(parameter_name + " by epoch")

                plt.savefig("debug/" + parameter_name  + ".png")
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
                plt.title(parameter_name + " by epoch")

                plt.subplot(212)
                plt.plot(I, dP, "b-")
                plt.xlabel("Epoch")
                plt.ylabel(d_parameter_name)

                plt.savefig("debug/" + parameter_name  + ".png")
                plt.close()

        return J[0][0]


    def predict(self, X, W = None, b = None, return_hidden = False):
        if(W == None):
            W = self.W

        if(b == None):
            b = self.b

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
