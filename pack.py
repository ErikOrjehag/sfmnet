import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Train and eval pack+unpack.')
parser.add_argument('--do', default="", type=str, 
                    help='What to do.')
args = parser.parse_args()
print('\nCurrent arguments -> ', args, '\n')

class DepthToSpace(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    self.bs = block_size
  
  def forward(self, x):
    N, C, H, W = x.size()
    # (N, bs, bs, C//bs^2, H, W)
    x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
    # (N, C//bs^2, H, bs, W, bs)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    # (N, C//bs^2, H * bs, W * bs)
    x = x.view(N, C // (self.bs**2), H * self.bs, W * self.bs)
    return x

class SpaceToDepth(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    self.bs = block_size

  def forward(self, x):
    N, C, H, W = x.size()
    # (N, C, H//bs, bs, W//bs, bs)
    x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
    # (N, bs, bs, C, H//bs, W//bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    # (N, C*bs^2, H//bs, W//bs)
    x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)
    return x

class Pack(nn.Module):
  def __init__(self, R=2, D=2, K=3, Co=4):
    super().__init__()
    self.R = R
    self.s2d = SpaceToDepth(R)
    self.conv3d = nn.Conv3d(1, D, K, padding=1)
    self.conv2d = nn.Conv2d(R**2 * 3 * D, Co, K, padding=1)

  def forward(self, x):
    N, C, H, W = x.size()
    x = self.s2d(x)
    x = x.unsqueeze(1)
    x = self.conv3d(x)
    x = x.view(N, -1, H // self.R, W // self.R)
    x = self.conv2d(x)
    return x

class Unpack(nn.Module):
  def __init__(self, R=2, D=2, K=3, Co=4):
    super().__init__()
    self.R = R
    self.conv2d = nn.Conv2d(Co, R*R*3//D, K, padding=1)
    self.conv3d = nn.Conv3d(1, D, K, padding=1)
    self.d2s = DepthToSpace(R)

  def forward(self, x):
    N, C, H, W = x.size()
    x = self.conv2d(x)
    x = x.unsqueeze(1)
    x = self.conv3d(x)
    x = x.view(N, -1, H, W)
    x = self.d2s(x)
    return x

class PackUnpack(nn.Module):
  def __init__(self):
    super().__init__()
    self.pack = Pack()
    self.unpack = Unpack()

  def forward(self, x):
    x = self.pack(x)
    x = self.unpack(x)
    return x

def imshow(name, img, scale=1):
  img = img / 2 + 0.5
  npimg = img.numpy()
  npimg = np.transpose(npimg, (1, 2, 0))
  #npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
  npimg = npimg[:,:,::-1]
  npimg = cv2.resize(npimg, None, fx=scale, fy=scale, 
    interpolation=cv2.INTER_NEAREST)
  cv2.imshow(name, npimg)

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

PATH = './pack_net.pth'

if args.do == "spacedepth":

  with torch.no_grad():
    x = torch.randn(1, 3, 32, 32)
    y = SpaceToDepth(2)(x)
    z = DepthToSpace(2)(y)
    print("Original", x.shape)
    print("SpaceToDepth", y.shape)
    print("DepthToSpace", z.shape)
    imshow("SpaceToDepth then DepthToSpace", 
      torchvision.utils.make_grid([
        x.squeeze(), 
        z.squeeze()]), scale=6)
    cv2.waitKey(0)

elif args.do == "packunpack":

  with torch.no_grad():
    x = torch.randn(1, 3, 32, 32)
    y = Pack()(x)
    z = Unpack()(y)
    print("Original", x.shape)
    print("Pack", y.shape)
    print("Unpack", z.shape)
    imshow("Intermediate", 
      y.squeeze(), scale=6)
    imshow("Pack then Unpack", 
      torchvision.utils.make_grid([
        x.squeeze(),
        z.squeeze()]), scale=6)
    cv2.waitKey(0)

elif args.do == "train":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  net = PackUnpack()
  net.to(device)
  criterion = nn.L1Loss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, _ = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, inputs)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  print('Finished training')

  torch.save(net.state_dict(), PATH)

elif args.do == "test":
  net = PackUnpack()
  print("Parameters: %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))

  with torch.no_grad():
    dataiter = iter(testloader)

    for _ in range(10):
      inputs, _ = dataiter.next()

      net.load_state_dict(torch.load(PATH))
      outputs = net(inputs)

      imshow("Result", torchvision.utils.make_grid(
        torch.cat((inputs, outputs), 0), nrow=4), scale=6)
      cv2.waitKey(0)