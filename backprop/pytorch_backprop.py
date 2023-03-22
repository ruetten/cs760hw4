import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
import numpy as np

# This device thing was something I was trying to make my code run faster, not sure if it worked
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gather the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# Perform some type-casting gymnastics in order to get X and y into the forms that I want
X_train = torch.FloatTensor(np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_trainset])).to(device)
y_train =  torch.LongTensor(np.array([np.array(datapoint[1]) for datapoint in mnist_trainset])).to(device)

X_test = torch.FloatTensor(np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_testset])).to(device)
y_test = torch.LongTensor(np.array([np.array(datapoint[1]) for datapoint in mnist_testset])).to(device)

# Define a network in PyTorch
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(784, 300)
        self.layer2 = torch.nn.Linear(300, 200)
        self.layer3 = torch.nn.Linear(200, 10)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

net = Net().to(device)

# Cross Entropy Loss function and SGD and learning rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# Some hyperparameters
batch_size = 64
num_epochs = 50

# Store losses and accuracies for plotting purposes
losses = np.zeros(num_epochs)
accuracies = np.zeros(num_epochs)

# Train the network
for epoch in range(num_epochs):
  # Shuffle the training data
  indices = torch.randperm(X_train.shape[0])
  X_train = X_train[indices]
  y_train = y_train[indices]

  # For accuracy calculation
  correct = 0

  # Mini-batch gradient descent
  for i in range(0, X_train.shape[0], batch_size):
    # Get a mini-batch of data
    inputs = X_train[i:i+batch_size]
    labels = y_train[i:i+batch_size]

    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # For accuracy calculation
    yhat = np.zeros(labels.shape)
    for i in range(len(outputs)):
      yhat[i] = torch.argmax(outputs[i])
    correct = correct + np.count_nonzero(yhat == labels.numpy())

  # Accuracy calculation
  accuracy = 100 * correct / len(X_train)
  accuracies[epoch] = accuracy

  # Get current loss
  losses[epoch] = loss.item()

  # Print the loss every epoch
  print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, loss.item(), accuracy))

# Print out final accruacy
with torch.no_grad():
    outputs = net(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).float().mean()
    print('Accuracy on the test set: ', accuracy)

# Print losses and accuracies which I copy from the console to put into a matplotlib py plot
# (I tried to do it right in here but my computer was having complaints about matplotlib, so I did it elsewhere on Google Collab)
print('losses = [', end='')
for loss in losses:
    print(loss, end=', ')
print(']')
print('accuracies = [', end='')
for acc in accuracies:
    print(acc, end=', ')
print(']')

# import matplotlib.pyplot as plt
#
# plt.plot( losses)
# plt.title('Learning curve: Loss over time')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.show()
#
# plt.plot( accuracies)
# plt.title('Accuracy on test-set over time')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.show()
