import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.rev_head = nn.Linear(16, 1)
        self.surv_head = nn.Linear(16, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.shared(x)
        rev = self.rev_head(x)
        surv = self.sigmoid(self.surv_head(x))
        return rev, surv