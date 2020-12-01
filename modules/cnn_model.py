import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, groups):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 6)
        self.conv_layer2 = self._conv_layer_set(6, 12)
        self.fc1 = nn.Linear(26364, 100)
        self.fc2 = nn.Linear(100, groups)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(100)
        self.drop = nn.Dropout(p=0.50)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(4, 4, 4), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)))

        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out
