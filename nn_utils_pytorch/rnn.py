# Author: acxz
# Date: 11/2/19
# Brief: Class and methods for using an RNN

# Reference:
# https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py

import argparse
import functools
import pickle
import torch
import sys

# Define structure of model
class ModelRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ModelRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # FIXME how to specify a mode?
        #self.rnn = torch.nn.RNN('RNN_RELU', self.input_size,
        #        self.hidden_size, self.num_layers)
        self.rnn = torch.nn.RNN(self.input_size, self.hidden_size,
                self.num_layers, batch_first=True)

    def forward(self, input_, hx):
        return self.rnn.forward(input_, hx)

    # Method to train model
    # TODO add time predictions
    # FIXME maybe easy way to reduce duplicate code in train and test
    # FIXME should epoch be displayed here or outside
    def train(self, device, train_dataloader, optimizer, loss_func,
            accuracy_func, display_status, current_epoch, final_epoch,
            batch_display_interval):

        # Set model in training mode
        self.train()

        # Constants
        total_batches = len(train_dataloader)
        samples_per_batch = len(train_dataloader.dataset)
        total_samples = total_batches * samples_per_batch

        # Metrics to keep track of throughout batches
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

            # Split up our input_data into model input and initial hidden value
            # Also how to get initial hidden value into this method
            # TODO
            #initial_hidden_value = torch.randn(num_layers, batch_size, hidden_size)
            initial_hidden_value = None
            model_input = input_

            # TODO: what does this do?
            optimizer.zero_grad()

            # Predict
            output, final_hidden_value = self(model_input, initial_hidden_value)

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

    # Method to test model
    def test(self, device, test_dataloader, loss_func, accuracy_func,
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

                # Split up our input_data into model input and initial hidden value
                # Also how to get initial hidden value into this method
                # TODO
                #initial_hidden_value = torch.randn(num_layers, batch_size, hidden_size)
                #hidden_size = 5
                #model_input_size = input_.shape[2] - hidden_size
                #print(input_)
                #model_input, hidden_values = torch.split(input_, [model_input_size,
                #    hidden_size], dim=2)
                #print(model_input)
                #print(hidden_values)
                #initial_hidden_value = hidden_values[0]
                initial_hidden_value = None
                model_input = input_

                # Predict
                output, final_hidden_value = self(model_input, initial_hidden_value)

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

# Method to collate a batch of data
# Split up input samples into batches of sequences
# dim to be (batch, sequence, feature)
# TODO: incorporate, hidden layer stuff in here as well
def collate_func(total_samples, batch_size, sequence_size):

    # Go through each sample in total_samples to form a sequence
    for sample_num in range(0, len(total_samples)):
        sample = total_samples[sample_num]

        # Get input/output from each sample
        input_sample = sample[0]
        output_sample = sample[1]

        # Create input/output sequence
        if (sample_num == 0):
            input_sequence = input_sample
            output_sequence = output_sample
        # Append it to input/output sequence
        else:
            input_sequence = torch.cat((input_sequence, input_sample), dim=0)
            output_sequence = torch.cat((output_sequence, output_sample), dim=0)

    # Reshape appeneded features into (total_samples, features)
    input_sequence = torch.reshape(input_sequence, (len(total_samples),
        len(input_sample)))
    output_sequence = torch.reshape(output_sequence, (len(total_samples),
        len(output_sample)))

    # Reshape (total_samples, features) into (batch, sequence, feature)
    input_batch = torch.reshape(input_sequence, (batch_size, sequence_size,
        len(input_sample)))
    output_batch = torch.reshape(output_sequence, (batch_size, sequence_size,
        len(output_sample)))

    return input_batch, output_batch


## Task specific functions

# Method to evaluate accuracy
# FIXME
def check_accuracy(vector_1, vector_2):
    is_accurate = False
    if (torch.allclose(vector_1, vector_2, rtol=1e-05, atol=1e-08)):
        is_accurate = True
    return is_accurate

# Main method to read in data and train/test/save model
# TODO: Cleanup and argparse?
""" Because some syntax error for a simple if statement ugh
def main():

    training_dataset_filename = "../tab-parser/Flightlab_Flight_Data/19NOV18_Task10_Continous_flight10.tab.io.pkl"
    testing_dataset_filename = "../tab-parser/Flightlab_Flight_Data/19NOV18_Task10_Continous_flight11.tab.io.pkl"
    train_batch_size = 5
    test_batch_size = 5
    sequence_size = 4
    epochs = 2
    lr = 0.01
    no_cuda = False
    display_status = True
    epoch_display_interval = 1
    batch_display_interval = 2
    use_model = ""
    #use_model = "test_rnn.pt"
    save_model = "test_rnn.pt"

    # Setup
    seed = 1
    torch.manual_seed(seed)

    # Use CUDA if specified
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Read in training dataset
    training_dataframe = pickle.load(open(training_dataset_filename, 'rb'))
    training_dataset_input = training_dataframe["INPUT"].values[0:50]
    training_dataset_output = training_dataframe["OUTPUT"].values[0:50]

    training_dataset = CustomDataset(training_dataset_input,
            training_dataset_output)

    # Initialize collate_func for training data
    train_collate_func = functools.partial(collate_func, batch_size=train_batch_size,
            sequence_size=sequence_size)
    train_dataloader = torch.utils.data.DataLoader(dataset=training_dataset,
        # pseudo batch_size so that our traning batch size is given argument
        # divided by our sequence_size
        batch_size=train_batch_size * sequence_size,
        sampler=torch.utils.data.SequentialSampler(training_dataset),
        collate_fn=train_collate_func,
        drop_last=True, **kwargs)

    # Read in testing dataset
    testing_dataframe = pickle.load(open(testing_dataset_filename, 'rb'))
    testing_dataset_input = testing_dataframe["INPUT"].values[0:50]
    testing_dataset_output = testing_dataframe["OUTPUT"].values[0:50]

    testing_dataset = CustomDataset(testing_dataset_input,
            testing_dataset_output)

    # Initialize collate_func for testing data
    test_collate_func = functools.partial(collate_func, batch_size=test_batch_size,
            sequence_size=sequence_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset,
        # pseudo batch_size so that our traning batch size is given argument
        # divided by our sequence_size
        batch_size=test_batch_size * sequence_size,
        sampler=torch.utils.data.SequentialSampler(testing_dataset),
        collate_fn=test_collate_func,
        drop_last=True, **kwargs)

    # Parameters to initialize model
    input_dim = len(training_dataset_input[0])
    output_dim = len(training_dataset_output[0])

    #input_dim = 3
    #output_dim = 5

    input_size = input_dim
    hidden_size = output_dim
    num_layers = 1

    # Initialize model
    model = ModelRNN(input_size, hidden_size, num_layers).to(device)
    if (use_model != ""):
        model.load_state_dict(torch.load(use_model), strict=False)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr)

    # Initialize loss function
    loss_func = torch.nn.MSELoss()

    # Initialize accuracy function
    accuracy_func = check_accuracy

    ## Train the model
    for epoch in range(0, epochs):
        if (epoch % epoch_display_interval == 0):
            epoch_display_status = True
        else:
            epoch_display_status = False

        model.train(device, train_dataloader, optimizer, loss_func,
        #        accuracy_func, epoch_display_status, epoch, epochs,
        #        batch_display_interval)
        model.test(device, test_dataloader, loss_func, accuracy_func,
                epoch_display_status, epoch, epochs, batch_display_interval)

    # Let us just test some predictions
    # (batch, seq, feature)
    #input_ = torch.randn(batch_size, sequence_size, input_size)
    #h0 = torch.randn(num_layers, batch_size, hidden_size)
    #my_output, my_hn = model(input_.to(device), h0.to(device))

    #print("input: ")
    #print(input_)

    #print("h0: ")
    #print(h0)

    #print("my output: ")
    #print(my_output)
    #print("my hn: ")
    #print(my_hn)

    # Save the model
    if (save_model != ""):
        torch.save(model.state_dict(), save_model)


if __name__ == "__main__":
    main()
"""
