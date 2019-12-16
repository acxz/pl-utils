# Author: acxz
# Date: 9/27/19
# Brief: Generate a continous dynamics model

import sys
import pickle
import torch
import argparse

# Define Dataset subclass to facilitate batch training
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Define structure of NN
class Net(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hl1_size, hl2_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hl1_size)
        self.fc2 = torch.nn.Linear(hl1_size, hl2_size)
        self.fc3 = torch.nn.Linear(hl2_size, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        loss = torch.nn.functional.mse_loss(output, target.float())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            # sum up batch loss
            test_loss += torch.nn.functional.mse_loss(
                output, target.float(), reduction='sum').item()
            # Measure accuracy
            pred = output
            if (torch.allclose(pred, target.float(), atol = float(0.001))):
                correct += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Dynamics Model')
    parser.add_argument(
        "--training-dataset",
        type=str,
        help="training dataset file path")
    parser.add_argument(
        "--testing-dataset",
        type=str,
        help="testing dataset file path")
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1,
        metavar='N',
        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--h1_size', type=int, default=32, metavar='H1',
            help='size of hidden layer 1 (default: 32')
    parser.add_argument('--h2_size', type=int, default=32, metavar='H2',
            help='size of hidden layer 2 (default: 32')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str, default="",
                        help='For Saving the current Model')
    parser.add_argument('--use-model', type=str,
                        default="", help='For using a saved model')

    args = parser.parse_args()

    # TODO: Why is this here?
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Read in training dataset
    training_dataframe = pickle.load(open(args.training_dataset, 'rb'))
    training_dataset_input = training_dataframe["INPUT"].values
    training_dataset_output = training_dataframe["OUTPUT"].values

    # Read in testing dataset
    testing_dataframe = pickle.load(open(args.testing_dataset, 'rb'))
    testing_dataset_input = testing_dataframe["INPUT"].values
    testing_dataset_output = testing_dataframe["OUTPUT"].values

    train_loader = torch.utils.data.DataLoader(dataset=SimpleDataset(
        training_dataset_input, training_dataset_output),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=SimpleDataset(
        testing_dataset_input, testing_dataset_output),
        batch_size = args.batch_size, shuffle = True, **kwargs)

    input_dim=len(training_dataset_input[0])
    output_dim=len(training_dataset_output[0])
    print(input_dim)
    print(output_dim)

    # Initialize model
    model=Net(input_dim, output_dim, args.h1_size, args.h2_size).to(device)
    if (args.use_model != ""):
        model.load_state_dict(torch.load(args.use_model), strict=False)

    # Initialize optimizer
    optimizer=torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model != ""):
        torch.save(model.state_dict(), args.save_model)


if __name__ == "__main__":
    main()
