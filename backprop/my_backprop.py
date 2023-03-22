import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt

# Get the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# Perform some type gymnastics to get the data into the format that I want
X_train = np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_trainset])
y_train = np.array([np.array(datapoint[1]) for datapoint in mnist_trainset])

X_test = np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_testset])
y_test = np.array([np.array(datapoint[1]) for datapoint in mnist_testset])

X = X_train
y = y_train.reshape((len(y_train), 1))

# Define dimensions of my weights and such
d = X.shape[1]
d1 = 300
d2 = 200
k = len(np.unique(y))

W1 = np.random.random((d1, d)) * 2 - 1
W2 = np.random.random((d2, d1)) * 2 - 1
W3 = np.random.random((k, d2)) * 2 - 1

#print(X.shape, y.shape, W1.shape, W2.shape, W3.shape)

# Functions
def sigmoid(z):
  return 1 / (1 + np.exp(z))

def softmax(z):
 e = np.exp(z)
 return e / e.sum()

def vectorize_y(yi):
  z = np.zeros(k)
  z[int(yi)] = 1
  return z

def feedforwardnetwork(X, W1, W2, W3):
  z1 = np.matmul(W1, X.T).T
  a1 = sigmoid(z1)
  z2 = np.matmul(W2, a1.T).T
  a2 = sigmoid(z2)
  z3 = np.matmul(W3, a2.T).T

  yhat = np.zeros((z3.shape))
  for z3i in range(len(z3)):
    yhat[z3i] = softmax(z3[z3i])

  return a1, a2, yhat

# hyper-parameters
α = 0.005
batch_size = 8

# Train network
iterations = 10000
accuracies = np.zeros(int(iterations / 100))
losses = np.zeros(int(iterations / 100))

for itr in range(iterations):
  idx = np.random.randint(0, len(X), size=batch_size)

  # Feed forward to get yhat and one-hot version of y
  a1, a2, yhat = feedforwardnetwork(X[idx], W1, W2, W3)
  y_true = np.array([vectorize_y(yi) for yi in y[idx]])

  # Third layer back prop
  delta3 = yhat - y_true
  dL_dW3 = (delta3.T @ a2) / batch_size # divide by batch size in order to get average over all examples

  # Second layer back prop
  delta2 = (delta3 @ W3) * (a2 * (1 - a2))
  dL_dW2 = (delta2.T @ a1) / batch_size

  # First layer back prop
  delta1 = (delta2 @ W2) * (a1 * (1 - a1))
  dL_dW1 = (delta1.T @ X[idx]) / batch_size

  # Weight updates
  W3 = W3 - α * dL_dW3
  W2 = W2 - α * dL_dW2
  W1 = W1 - α * dL_dW1

  # Every 100 iterations, print out current stats:
  if itr % 100 == 0:
    L = np.average([-np.sum(loss) for loss in (y_true*np.log(yhat))])
    losses[int(itr / 100)] = L
    print('Loss:', L, end=' \t')

    a1, a2, yhat = feedforwardnetwork(X_test, W1, W2, W3)
    prediction = np.array([np.argmax(yhat_i) for yhat_i in yhat])
    acc = np.count_nonzero(prediction == y_test.T) / len(X_test)
    print('Acc on test:', acc)
    accuracies[int(itr / 100)] = acc

# Test network on test data
a1, a2, yhat = feedforwardnetwork(X_test, W1, W2, W3)
prediction = np.array([np.argmax(yhat_i) for yhat_i in yhat])
print(np.count_nonzero(prediction == y_test.T) / len(X_test))

# plt.plot(losses)
# plt.title('Learning curve: Loss over time')
# plt.ylabel('Loss')
# plt.xlabel('Iteration (100s)')
# plt.show()
#
# plt.plot(accuracies)
# plt.title('Accuracy on test-set over time')
# plt.ylabel('Accuracy')
# plt.xlabel('Iteration (100s)')
# plt.show()


### My implementation that I attempted with element-wise operations
### Ended up being really slow and no different performance than the other
# W1 = np.random.random((d1, d)) * 2 - 1
# W2 = np.random.random((d2, d1)) * 2 - 1
# W3 = np.random.random((k, d2)) * 2 - 1
# for iteration in range(0, 5000):
#   idx = np.random.randint(0, len(X), size=batch_size)
#
#   a1, a2, yhat = feedforwardnetwork(X[idx], W1, W2, W3)
#   y_true = np.array([vectorize_y(yi) for yi in y[idx]])
#
#   # Third layer
#   delta3 = yhat - y_true
#
#   dL_dW3 = np.zeros((k, d2))
#   for i in range(k):
#     for j in range(d2):
#       dL_dW3[i, j] = delta3[0][i] * a2[0][j]
#
#   # Second layer
#   temp = np.zeros((k, d2))
#   for i in range(k):
#     for j in range(d2):
#       temp[i][j] =  W3[i][j] * delta3[0][i]
#
#   sum = np.zeros((1, d2))
#   for n in range(k):
#     sum = sum + temp[n]
#
#   delta2 = sum * (a2 * (1 - a2))
#
#   dL_dW2 = np.zeros((d2, d1))
#   for i in range(d2):
#     for j in range(d1):
#       dL_dW2[i, j] = delta2[0][i] * a1[0][j]
#
#   # First layer
#   temp = np.zeros((d2, d1))
#   for i in range(d2):
#     for j in range(d1):
#       temp[i][j] =  W2[i][j] * delta2[0][i]
#
#   sum = np.zeros((1, d1))
#   for n in range(d2):
#     sum = sum + temp[n]
#
#   delta1 = sum * (a1 * (1 - a1))
#
#   dL_dW1 = np.zeros((d1, d))
#   for i in range(d1):
#     for j in range(d):
#       dL_dW1[i, j] = delta1[0][i] * X[0][j]
#
#   W3 = W3 - α * dL_dW3
#   W2 = W2 - α * dL_dW2
#   W1 = W1 - α * dL_dW1
#
#   if iteration % 10 == 0:
#     #L = -np.sum(y_true*np.log(yhat))
#     L = np.average([-np.sum(loss) for loss in (y_true*np.log(yhat))])
#     a1, a2, yhat = feedforwardnetwork(X[:100], W1, W2, W3)
#     prediction = np.array([np.argmax(yhat_i) for yhat_i in yhat])
#     #print(prediction ==  y[:100])
#     # print(prediction)
#     # print(y[:100].T)
#     print(L, end=' ')
#     print(np.count_nonzero(prediction == y[:100].T) / 100)
