import nnl
import numpy as np
import matplotlib.pyplot as plt

#X, Y = nnl.linear_data()
#X, Y = nnl.non_linear_data()
X, Y = nnl.very_wiggly_data()

X_norm = nnl.normalize(X)
Y_norm = nnl.normalize(Y)

# Linear function (ca. 1s to train)
#nn = nnl.NN(depth = 1, width = 2)

#training_mse = nn.train(X_norm, Y_norm, learning_rate = 0.12, epochs = 100)
#training_mse = nn.train(X_norm, Y_norm, learning_rate = 0.12, epochs = 100, batch_size = 2)

# Non-linear function (ca. 4s to train)
#nn = nnl.NN(depth = 3, width = 5)

#training_mse = nn.train(X_norm, Y_norm, learning_rate = 0.025, epochs = 3000)
#training_mse = nn.train(X_norm, Y_norm, learning_rate = 0.1, epochs = 3000, batch_size = 16)

# Very wiggly function (ca. 16 mins to train)
nn = nnl.NN(depth = 5, width = 10)

training_mse = nn.train(X_norm, Y_norm, learning_rate = 0.01, epochs = 30000, printing = True)
#training_mse = nn.train(X_norm, Y_norm, learning_rate = 0.03, epochs = 30000, batch_size = 128, printing = True)

Y_hat_norm = nn.predict(X_norm)

print("Training MSE: " + str(training_mse))

mean_Y = np.mean(Y)
sd_Y = np.std(Y)
Y_hat = sd_Y * Y_hat_norm + mean_Y

plt.plot(X[0], Y[0], "o")
plt.plot(X[0], Y_hat[0], "r-")
plt.show()
