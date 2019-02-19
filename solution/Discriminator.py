import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_dim=28*28):
        super(Discriminator, self).__init__()
        # TODO: Complete the model architecture
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            self._modules[m].weight.data.normal_(mean, std)
            self._modules[m].bias.data.zero_()

    def forward(self, x):
        # TODO: Complete the forward pass
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
