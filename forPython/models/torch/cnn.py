from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleCNN(nn.Module):
    def __init__(self, n_features, mid_size, n_outputs):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(mid_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        """
        :type x: torch.Tensor
        """
        x = x.transpose(1, 2)
        outputs = self.conv(x)
        outputs = self.fc(outputs)
        return outputs
