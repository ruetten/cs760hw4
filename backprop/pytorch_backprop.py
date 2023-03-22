import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

X_train = torch.FloatTensor(np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_trainset])).to(device)
y_train =  torch.LongTensor(np.array([np.array(datapoint[1]) for datapoint in mnist_trainset])).to(device)

X_test = torch.FloatTensor(np.array([np.array(datapoint[0]).reshape(784) for datapoint in mnist_testset])).to(device)
y_test = torch.LongTensor(np.array([np.array(datapoint[1]) for datapoint in mnist_testset])).to(device)

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

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

batch_size = 10
num_epochs = 50

for epoch in range(num_epochs):
  # Shuffle the training data
  indices = torch.randperm(X_train.shape[0])
  X_train = X_train[indices]
  y_train = y_train[indices]

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

  # Print the loss every epoch
  print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


with torch.no_grad():
    outputs = net(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).float().mean()
    print('Accuracy on the test set: ', accuracy)
