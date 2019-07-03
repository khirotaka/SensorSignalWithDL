import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from forPython.datasets.uci import load_mhealth
from forPython.models.torch.cnn import SimpleCNN
from forPython.utility.trainer import TorchSimpleTrainer

np.random.seed(0)
torch.random.manual_seed(0)


(x_train, y_train), (x_test, y_test) = load_mhealth()
y_train -= 1
y_test -= 1

n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 12
batch_size, epochs = 32, 10

x_train = torch.tensor(x_train).float()
x_test = torch.tensor(x_test).float()

y_train = torch.tensor(y_train[:, 0]).long()
y_test = torch.tensor(y_test[:, 0]).long()

mid_size = 128 * 62

model = SimpleCNN(n_features, mid_size, n_outputs)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size, False)
test_loader = DataLoader(test_ds, batch_size, False)

clf = TorchSimpleTrainer(model, loss_func, optimizer)
clf.fit(train_loader, epochs)
clf.evaluate(test_loader)
