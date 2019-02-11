import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from data_loader import BPDataSet
from unet import UNet

train_dataloader = torch.utils.data.DataLoader(BPDataSet(), batch_size=1, shuffle=True)

device = torch.device('cpu')
model = UNet(1, 1)
model = model.to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 3

for epoch in range(0, n_epochs):
    train_loss = 0.0

    model.train()
    for data, target in train_dataloader:
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*data.size(0)

        print(train_loss)


torch.save(model.state_dict(), 'bp_model.pt')

