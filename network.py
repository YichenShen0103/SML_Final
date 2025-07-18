import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size=[32, 16]):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

    def forward(self, x):
        return self.net(x)
