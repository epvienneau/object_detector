from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from img_loader import img_loader
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

training_loss = []
validation_loss = []

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target= data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        training_loss.append(loss.item())
  
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.MSELoss()
            test_loss += criterion(output, target).item() 
            #pred = output.max(1, keepdim=True)[1]
            correct += output.eq(target.view_as(output)).sum().item()

    test_loss /= len(test_loader.dataset)
    validation_loss.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # parse txt file to create dictionary for dataloader
    coord1_train = []
    coord2_train = []
    coord1_val = []
    coord2_val = []
    img_file_train = []
    img_file_val = []
    img_path_train = ['train/']*105
    img_path_val = ['validation/']*10
    with open('labels/labels.txt', 'r') as f:
        for count, line in enumerate(f):
            #line = f.readline()
            line = line.split() #now line is a list
            file_num = line[0]
            file_num = file_num.split('.')
            file_num = int(file_num[0]) #now file_num is just the file number 
            if file_num<111:
                img_file_train.append(line[0])
                coord1_train.append(float(line[1]))
                coord2_train.append(float(line[2]))
            else:
                img_file_val.append(line[0])
                coord1_val.append(float(line[1]))
                coord2_val.append(float(line[2]))
    data_train = [coord1_train, coord2_train, img_file_train, img_path_train]
    data_val = [coord1_val, coord2_val, img_file_val, img_path_val] 
    train_loader = torch.utils.data.DataLoader(img_loader(data_train), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(img_loader(data_val), batch_size=args.batch_size, shuffle=True, **kwargs) 

    model = models.resnet18(pretrained=True, **kwargs).to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.double()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    #epoch_axis = range(args.epochs)
    #plt.plot(epoch_axis, training_loss, 'r', epoch_axis, validation_loss, 'b')
    #plt.show()
    torch.save(model.state_dict(), './Resnetmodel.pt')
    with open('loss.csv', 'w', newline='') as csvfile:
        losswriter = csv.writer(csvfile, delimiter=' ', 
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        losswriter.writerow('training')
        print(len(training_loss))
        print(len(validation_loss))
        for item in training_loss:
            losswriter.writerow(str(round(item, 4)))
        losswriter.writerow('validation')
        for item in validation_loss:
            losswriter.writerow(str(round(item, 4)))


if __name__ == '__main__':
    main()

