from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# The lines above are import statements, which make the necessary packages available for use in this program.
# The "as" phrases allow "nicknames" for the packages to be established.
# The first line makes print a function.

class Net(nn.Module):
    # This defines a new class, Net, that accepts as an input parameter a pytorch neural network module. Now that this
    # is defined, new objects of class Net can be instantiated.
    def __init__(self):
        # This is the constructor method for the class Net, which defines how all the attributes of a new Net object
        # should be initialized when one is instantiated. The parameter self is a variable pointing to the current
        # object (analogous to "this"). Self must be explicitly defined in the method definitions.
        super(Net, self).__init__()
        # The above line allows for multiple inheritance from the parent classes of Net
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Define a 1D convolution attribute of the Net
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Define a 2D convolution attribute of the Net
        self.conv2_drop = nn.Dropout2d()
        # Define a drop-out convolution attribute of the net
        self.fc1 = nn.Linear(320, 50)
        # Define a linear transformation attribute of the Net
        self.fc2 = nn.Linear(50, 10)
        # Define another linear transformation attribute of the Net

    def forward(self, x):
        # A method called forward that takes as its input the state of the neural net at one time point and returns
        # the state of the neural net at the next time point (or "step forward") in the iterations.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Convolution, pooling, and relu activation function
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Another convolution, drop out convolution, pooling, relu activation function
        x = x.view(-1, 320)
        # View data?
        x = F.relu(self.fc1(x))
        # Do a linear transformation then relu activation function
        x = F.dropout(x, training=self.training)
        # Do the training step (the actual "step" forward), with dropout
        x = self.fc2(x)
        # Do another linear transformation
        return F.log_softmax(x, dim=1) # return the softmax to put value between 0 and 1

def train(args, model, device, train_loader, optimizer, epoch):
    # The training step that must occur each iteration.
    # Dataset is divided into batches, where only the images in a batch are processed together.
    # Entire dataset (all the batches) are processed several times, which is the number of epochs.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    # This function assesses how well the neural net is doing in terms of accuracy.
    # A loss function is defined and evaluated to measure the success of the net at the point in time during which
    # this method is called.
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # This is the function that runs when this script is called. Rather than defining new methods,
    # it calls the methods previously defined or imported to produce the desired output, which is in this case is
    # real-time feedback about how the model is doing as it is being trained.
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    # Here, a new object of class Net named model is instantiated as defined above.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # In this for loop, the train and test methods previously defined in this script are called to train each batch
        # for the current epoch and then see how well the model is doing. This repeats until all the epochs are finished.
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

# When this python script, main.py, is called, the method called main() (above) runs.
if __name__ == '__main__':
    main()

