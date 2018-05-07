'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import *
from utils import progress_bar


# Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD momentum')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batchsize', '-b', default=128, type=int,
                    help='Training batch size')
parser.add_argument('--maxbatches', '-B', default=None, type=int,
                    help='Max number of batches per epoch')
parser.add_argument('--model', '-m', default='VGG16', type=str,
                    help='Model name')
parser.add_argument('--optimizer', '-o', default='SGD', type=str,
                    help='Optimization Algorithm')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

# Model
model_name = args.model
models = {
    'VGG16': lambda: VGG('VGG16'),
    'ResNet18': lambda: ResNet18(),
    'PreActResNet18': lambda: PreActResNet18(),
    'GoogLeNet': lambda: GoogLeNet(),
    'DenseNet121': lambda: DenseNet121(),
    'ResNeXt29_2x64d': lambda: ResNeXt29_2x64d(),
    'MobileNet': lambda: MobileNet(),
    'MobileNetV2': lambda: MobileNetV2(),
    'DPN92': lambda: DPN92(),
    'ShuffleNetG2': lambda: ShuffleNetG2(),
    'SENet18': lambda: SENet18(),
}

print('==> Building model..')
if model_name not in models:
    raise ValueError("Invalid model name %s" % model_name)

net = models[model_name]()
net = net.to(device)
if device == 'cuda':
    cuda_device_count = torch.cuda.device_count()
    print("Using %d GPUs" % cuda_device_count)
    if cuda_device_count > 1:
        # DataParallel causes pytorch to use multiple GPUs
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print(net)

checkpoint_filename = './checkpoint/ckpt.%s.t7' % model_name
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint %s..' % checkpoint_filename)
    assert os.path.isfile(checkpoint_filename), 'Error: no checkpoint found!'
    checkpoint = torch.load(checkpoint_filename)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss
criterion = nn.CrossEntropyLoss()

# Optimization algorithm
optimizers = {
    'SGD': lambda: optim.SGD(net.parameters(),
                             lr=args.lr,
                             momentum=args.momentum,
                             weight_decay=5e-4),
    'Adam': lambda: optim.Adam(net.parameters(), lr=args.lr),
}

optimizer_name = args.optimizer
if optimizer_name not in optimizers:
    raise ValueError("Invalid optimizer name %s" % optimizer_name)

print("==> Building optimizer %s.." % optimizer_name)
optimizer = optimizers[args.optimizer]()
print(optimizer)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    num_batches = len(trainloader)
    if args.maxbatches:
        num_batches = args.maxbatches

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, num_batches, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total,
                        correct, total))

        if batch_idx + 1 >= num_batches:
            break


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    num_batches = len(testloader)
    if args.maxbatches:
        num_batches = args.maxbatches

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, num_batches, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total,
                        correct, total))

        if batch_idx + 1 >= num_batches:
            break

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving %s..' % checkpoint_filename)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_filename)
        best_acc = acc





# Train the model
for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
