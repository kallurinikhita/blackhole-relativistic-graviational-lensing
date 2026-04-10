import torch

data = torch.load("dataset.pt", weights_only=False)

X = data["X"]   # shape: (2, 256, 256)
Y = data["Y"]   # shape: (2, 256, 256)

# add batch dimension
X = torch.tensor(X).float().unsqueeze(0)  # (1, 2, 256, 256)
Y = torch.tensor(Y).float().unsqueeze(0)  # (1, 2, 256, 256)


import torch.nn as nn

class LensingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 2, 3, padding=1)  # output: dx, dy
        )

    def forward(self, x):
        return self.model(x)
    


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(2, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)

        # Decoder
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 2, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):

        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))

        d1 = self.up(e2)
        d1 = torch.relu(self.dec1(d1))

        out = self.out(d1)

        return out