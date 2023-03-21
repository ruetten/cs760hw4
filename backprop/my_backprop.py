import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
import numpy as np

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

X_train = np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_trainset])
y_train = np.array([np.array(datapoint[1]) for datapoint in mnist_trainset])

X_test = np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_testset])
y_test = np.array([np.array(datapoint[1]) for datapoint in mnist_testset])

X = X_train
y = y_train.reshape((len(y_train), 1))

d = X.shape[1]
d1 = 300
d2 = 200
k = len(np.unique(y))

W1 = np.random.random((d1, d)) * 2 - 1
W2 = np.random.random((d2, d1)) * 2 - 1
W3 = np.random.random((k, d2)) * 2 - 1

X.shape, y.shape, W1.shape, W2.shape, W3.shape

def sigmoid(z):
  return 1 / (1 + np.exp(z))

def softmax(z):
  return np.exp(z) / np.sum(np.exp(z))

def vectorize_y(yi):
  z = np.zeros(k)
  z[int(yi)] = 1
  return z

def feedforwardnetwork(X, W1, W2, W3):
  #print(W1.shape, X.shape)
  z1 = np.matmul(W1, X.T).T
  a1 = sigmoid(z1)
  #print(W2.shape, a1.shape)
  z2 = np.matmul(W2, a1.T).T
  a2 = sigmoid(z2)
  #print(W3.shape, a2.shape)
  z3 = np.matmul(W3, a2.T).T
  yhat = softmax(z3)
  return a1, a2, yhat

# hyper-parameters
α = 0.005
batch_size = 8

W1 = np.random.random((d1, d)) * 2 - 1
W2 = np.random.random((d2, d1)) * 2 - 1
W3 = np.random.random((k, d2)) * 2 - 1

for i in range(0, 50000):
  idx = np.random.randint(0, len(X), size=batch_size)

  a1, a2, yhat = feedforwardnetwork(X[idx], W1, W2, W3)
  y_true = np.array([vectorize_y(yi) for yi in y[idx]])

  delta3 = yhat - y_true
  dL_dW3 = (delta3.T @ a2) / batch_size

  delta2 = (delta3 @ W3) * (a2 * (1 - a2))
  dL_dW2 = (delta2.T @ a1) / batch_size

  delta1 = (delta2 @ W2) * (a1 * (1 - a1))
  dL_dW1 = (delta1.T @ X[idx]) / batch_size

  W3 = W3 - α * dL_dW3
  W2 = W2 - α * dL_dW2
  W1 = W1 - α * dL_dW1

  if i % 100 == 0:
    #L = -np.sum(y_true*np.log(yhat))
    L = np.average([-np.sum(loss) for loss in (y_true*np.log(yhat))])
    a1, a2, yhat = feedforwardnetwork(X[:100], W1, W2, W3)
    prediction = np.array([np.argmax(yhat_i) for yhat_i in yhat])
    #print(prediction ==  y[:100])
    print(prediction)
    print(y[:100].T)
    print(L, end=' ')
    print(np.count_nonzero(prediction == y[:100].T) / 100)

a1, a2, yhat = feedforwardnetwork(X,W1,W2, W3)
prediction = np.array([np.argmax(yhat_i) for yhat_i in yhat])
print(prediction)
print(y.T)
np.count_nonzero(prediction == y.T) / 100
