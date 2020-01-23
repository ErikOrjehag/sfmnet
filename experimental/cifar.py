import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import cv2
import numpy as np

booltype = lambda s: s in ["yes", "1", "true", "True"]
parser = argparse.ArgumentParser(description='Train and eval cifar.')
parser.add_argument('--train', default=False, type=booltype, 
                    help='Train the network.')
args = parser.parse_args()
print('\nCurrent arguments -> ', args, '\n')

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
  root='./data', 
  train=True, 
  download=True,
  transform=transform)

trainloader = torch.utils.data.DataLoader(
  trainset, 
  batch_size=4,
  shuffle=True,
  num_workers=2)

testset = torchvision.datasets.CIFAR10(
  root='./data',
  train=False,
  download=True,
  transform=transform)

testloader = torch.utils.data.DataLoader(
  testset,
  batch_size=4,
  shuffle=False,
  num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'hourse', 'ship', 'truck')

PATH = './cifar_net.pth'

def imshow(name, img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  cv2.imshow(name, np.transpose(npimg, (1, 2, 0)))

def train():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  net = Net()
  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  print('Finished training')

  torch.save(net.state_dict(), PATH)

def test():
  dataiter = iter(testloader)
  images, labels = dataiter.next()
  print('Ground truth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

  net = Net()
  net.load_state_dict(torch.load(PATH))

  outputs = net(images)
  _, predicted = torch.max(outputs, 1)
  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

  #imshow("input", torchvision.utils.make_grid(images))
  #cv2.waitKey(0)

  """
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
  """
  class_correct = list(range(len(classes)))
  class_total = list(range(len(classes)))
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(predicted.shape[0]):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
  for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
      classes[i], 100 * class_correct[i] / class_total[i]))

if args.train:
  train()
else:
  test()