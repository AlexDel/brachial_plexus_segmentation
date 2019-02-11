import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from data_loader import BPDataSet
from unet import ResNetUNet

train_dataloader = torch.utils.data.DataLoader(BPDataSet, batch_size=10, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(n_class=1)
model = model.to(device)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

criterion = CrossEntropyLoss2d()

optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 3

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_dataloader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)


torch.save(model.state_dict(), 'bp_model.pt')

