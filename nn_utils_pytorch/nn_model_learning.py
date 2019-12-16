#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Author: acxz
# Date: 10/15/19
# Brief: Generate a helicopter system model


# In[15]:


import pickle
import torch


# In[16]:


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
        #print("input " + str(x[0]))

        x = self.fc1(x)
        #print("l1 " + str(x[0]))
        x = torch.tanh(x)
        #print("act1 " + str(x[0]))

        x = self.fc2(x)
        #print("l2 " + str(x[0]))
        x = torch.tanh(x)
        #print("act2 " + str(x[0]))

        x = self.fc3(x)
        #print("output " + str(x[0]))

        return x


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        loss = torch.nn.functional.mse_loss(output, target.float())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
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
            #print(abs(pred - target.float()))
            if (torch.allclose(pred, target.float(), atol = float(1))):
                correct += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[24]:


# Input Parameters
def main():

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


    # In[25]:


    # Initialize model

    # TODO: Why is this here?
    torch.manual_seed(seed)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Read in training dataset
    training_dataframe = pickle.load(open(training_dataset, 'rb'))
    training_dataset_input = training_dataframe["INPUT"].values
    training_dataset_output = training_dataframe["OUTPUT"].values

    # Read in testing dataset
    testing_dataframe = pickle.load(open(testing_dataset, 'rb'))
    testing_dataset_input = testing_dataframe["INPUT"].values
    testing_dataset_output = testing_dataframe["OUTPUT"].values

    train_loader = torch.utils.data.DataLoader(dataset=SimpleDataset(
        training_dataset_input, training_dataset_output),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=SimpleDataset(
        testing_dataset_input, testing_dataset_output),
        batch_size=batch_size, shuffle=True, **kwargs)

    input_dim=len(training_dataset_input[0])
    output_dim=len(training_dataset_output[0])

    # Initialize model
    model = Net(input_dim, output_dim, h1_size, h2_size).to(device)
    if (use_model != ""):
        model.load_state_dict(torch.load(use_model), strict=False)

    # Initialize optimizer
    optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum)


    # In[26]:


    # Train the model
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)

    # Save the model

    if(save_model != ""):
        torch.save(model.state_dict(), save_model)


    # In[13]:





    # In[ ]:



if __name__ == "__main__":
    main()
