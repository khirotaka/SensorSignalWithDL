import sys
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader as BaseDataLoader
from torch.optim.optimizer import Optimizer as BaseOptimizer


class TorchBase:
    def __init__(self, model, loss_func, optimizer):
        """
        :type model: torch.nn.Module
        :type loss_func: torch.nn.modules.loss._Loss
        :type optimizer: BaseOptimizer
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer

    def fit(self, dataloader, epochs=1):
        """

        :type dataloader: BaseDataLoader
        """
        self.model.train()
        batch_size = dataloader.batch_size
        len_of_dataset = len(dataloader.dataset)

        for epoch in range(epochs):
            for step, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                sys.stdout.write(
                    "\rEpochs: {:03d}/{:03d} - {:03d}% ".format(
                        epoch + 1, epochs, int((batch_size * (step + 1) / len_of_dataset) * 100)
                    )
                )
                sys.stdout.flush()

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.loss_func(outputs, y)

                loss.backward()
                self.optimizer.step()
        sys.stdout.write("\n")

    def evaluate(self, dataloader):
        self.model.eval()

        running_loss = torch.tensor(0.0).float()
        running_corrects = torch.tensor(0).long()
        len_of_dataset = len(dataloader.dataset)

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)

                _, predictions = torch.max(outputs, 1)
                loss = self.loss_func(outputs, y)

                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(predictions == y.data)

        test_loss = running_loss / len_of_dataset
        test_acc = running_corrects.float() / len_of_dataset
        print("Loss: {:.3f} - Accuracy: {:.3%}".format(test_loss, test_acc))

    def predict(self, x):
        self.model.eval()

        if not torch.is_tensor(x):
            x = torch.tensor(x).float()

        with torch.no_grad():
            outputs = self.model(x)

        _, prediction = torch.max(outputs, 1)

        return prediction

    def predict_proba(self, x):
        self.model.eval()

        if not torch.is_tensor(x):
            x = torch.tensor(x).float()

        with torch.no_grad():
            outputs = self.model(x)

        percentage = torch.softmax(outputs, 1)

        return percentage
