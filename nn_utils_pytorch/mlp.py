# Author: acxz
# Date: 9/27/19
# Brief: Class and methods for using a MLP

import argparse
import functools
import pickle
import torch
import sys

# Define structure of model
class ModelMLP(torch.nn.Module):

    # TODO specify length and width of mlp via input vector tbh
    # Along with list of activation functions (prob just the actual functions
    # themselves
    def __init__(self, input_dim, output_dim, hl1_size, hl2_size):
        super(ModelMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hl1_size)
        self.fc2 = torch.nn.Linear(hl1_size, hl2_size)
        self.fc3 = torch.nn.Linear(hl2_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)

        return x

    # Method to train model
    # TODO add time predictions
    # FIXME maybe easy way to reduce duplicate code in train and test
    # FIXME should epoch be displayed here or outside
    def train_model(self, device, train_dataloader, optimizer, loss_func,
            accuracy_func, display_status, current_epoch, final_epoch,
            batch_display_interval):

        # Set model in training mode
        self.train()

        # Constants
        total_batches = len(train_dataloader)
        samples_per_batch = len(train_dataloader.dataset)
        total_samples = total_batches * samples_per_batch

        # Metrics to keep track throughout batches
        total_loss = 0
        total_num_correct = 0

        # Go through the training data one batch at a time
        for batch_num, (input_, target) in enumerate(train_dataloader):
            # Load data onto device
            input_, target = input_.to(device), target.to(device)

            # TODO: Is this needed
            # Make sure input_, target is in float
            input_ = input_.float()
            target = target.float()

            # TODO: what does this do?
            optimizer.zero_grad()

            # Predict
            output = self(input_)

            # Loss
            batch_loss = loss_func(output, target)

            # Backpropagate information and update gradients
            batch_loss.backward()
            optimizer.step()

            # Epoch status
            epoch_status = current_epoch / final_epoch * 100

            # Batch status
            batch_status = batch_num / total_batches * 100

            # Average batch loss
            total_loss = total_loss + batch_loss
            avg_batch_loss = batch_loss / total_batches

            # Batch accuracy
            accuracy_count = [1 for i in range(0, len(output)) if
                    accuracy_func(output[i], target[i]) == True]
            batch_num_correct = sum(accuracy_count)
            batch_accuracy = batch_num_correct / total_batches * 100
            total_num_correct = total_num_correct + batch_num_correct

            # Batch status information
            if (display_status == True):
                if (batch_num % batch_display_interval == 0):
                    batch_status_string = ('[Training] Epoch: {}/{} ({:.0f}%) \t Batch: '
                    '{}/{} ({:.0f}%) \t Average Batch Loss: {:.4f} \t Batch Accuracy: '
                    '{}/{} ({:.0f}%)').format(current_epoch, final_epoch,
                            epoch_status, batch_num, total_batches,
                            batch_status, avg_batch_loss, batch_num_correct,
                            samples_per_batch, batch_accuracy)

                    print(batch_status_string)

        # Epoch status
        current_epoch = current_epoch + 1
        epoch_status = current_epoch / final_epoch * 100

        # Average total loss
        avg_total_loss = total_loss / total_samples

        # Total accuracy
        total_accuracy = total_num_correct / total_samples

        # Epoch status information
        if (display_status == True):
            epoch_status_string = ('[Training] Epoch: {}/{} ({:.0f}%) \t Average Loss: {:.4f} \t Accuracy: '
            '{}/{} ({:.0f}%)\n').format(current_epoch, final_epoch, epoch_status,
                    avg_total_loss, total_num_correct, total_samples, total_accuracy)

            print(epoch_status_string)

    def test_model(self, device, test_dataloader, loss_func, accuracy_func,
            display_status, current_epoch, final_epoch, batch_display_interval):

        # Set model in testing mode
        self.eval()

        # Constants
        total_batches = len(test_dataloader)
        samples_per_batch = len(test_dataloader.dataset)
        total_samples = total_batches * samples_per_batch

        # Metrics to keep track of throughout batches
        total_loss = 0
        total_num_correct = 0

        # Don't need to worry about gradients when testing, only need it for
        # backprop during training
        with torch.no_grad():
            # Go through the testing data one batch at a time
            for batch_num, (input_, target) in enumerate(test_dataloader):
                # Load data onto device
                input_, target = input_.to(device), target.to(device)

                # Make sure input_, target is in float
                input_ = input_.float()
                target = target.float()

                # Predict
                output = self(input_)

                # Loss
                batch_loss = loss_func(output, target)
                total_loss = total_loss + batch_loss

                # Epoch status
                epoch_status = current_epoch / final_epoch * 100

                # Batch status
                batch_status = batch_num / total_batches * 100

                # Average batch loss
                avg_batch_loss = batch_loss / total_batches

                # Batch accuracy
                accuracy_count = [1 for i in range(0, len(output)) if
                        accuracy_func(output[i], target[i]) == True]
                batch_num_correct = sum(accuracy_count)
                batch_accuracy = batch_num_correct / total_batches * 100
                total_num_correct = total_num_correct + batch_num_correct

                # Batch status information
                if (display_status == True):
                    if (batch_num % batch_display_interval == 0):
                        batch_status_string = ('[Testing] Epoch: {}/{} ({:.0f}%) \t Batch: '
                        '{}/{} ({:.0f}%) \t Average Batch Loss: {:.4f} \t Batch Accuracy: '
                        '{}/{} ({:.0f}%)').format(current_epoch, final_epoch,
                                epoch_status, batch_num, total_batches,
                                batch_status, avg_batch_loss, batch_num_correct,
                                samples_per_batch, batch_accuracy)

                        print(batch_status_string)

            # Epoch status
            current_epoch = current_epoch + 1
            epoch_status = current_epoch / final_epoch * 100

            # Average total loss
            avg_total_loss = total_loss / total_samples

            # Total accuracy
            total_accuracy = total_num_correct / total_samples

            # Epoch status information
            if (display_status == True):
                epoch_status_string = ('[Testing] Epoch: {}/{} ({:.0f}%) \t Average Loss: {:.4f} \t Accuracy: '
                '{}/{} ({:.0f}%)\n').format(current_epoch, final_epoch,
                        epoch_status, avg_total_loss, total_num_correct,
                        total_samples, total_accuracy)

                print(epoch_status_string)

# Define a custom Dataset class to facilitate training
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Method to evaluate accuracy
# FIXME
def check_accuracy(vector_1, vector_2):
    is_accurate = False
    if (torch.allclose(vector_1, vector_2, rtol=1e-05, atol=1e-08)):
        is_accurate = True
    return is_accurate

# Main method to read in data and train/test/save model
# TODO: Cleanup and argparse?
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Multilayer Perceptron Model')
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

    training_dataset = "../tab-parser/Flightlab_Flight_Data/19NOV18_Task10_Continous_flight10.tab.io.pkl"
    testing_dataset = "../tab-parser/Flightlab_Flight_Data/19NOV18_Task10_Continous_flight11.tab.io.pkl"
    batch_size = 100
    test_batch_size = 1
    epochs = 10
    h1_size = 32
    h2_size = 32
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 1
    log_interval = 10
    save_model = "test.pt"
    use_model = "test.pt"

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

    train_dataloader = torch.utils.data.DataLoader(dataset=CustomDataset(
        training_dataset_input, training_dataset_output),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataloader = torch.utils.data.DataLoader(dataset=CustomDataset(
        testing_dataset_input, testing_dataset_output),
        batch_size = args.batch_size, shuffle = True, **kwargs)

    input_dim=len(training_dataset_input[0])
    output_dim=len(training_dataset_output[0])
    print(input_dim)
    print(output_dim)

    # Initialize model
    model=ModelMLP(input_dim, output_dim, args.h1_size, args.h2_size).to(device)
    if (args.use_model != ""):
        model.load_state_dict(torch.load(args.use_model), strict=False)

    # Initialize optimizer
    optimizer=torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train_model(device, train_dataloader, optimizer, epoch)
        model.test_model(device, test_dataloader)

    if (args.save_model != ""):
        torch.save(model.state_dict(), args.save_model)


if __name__ == "__main__":
    main()
