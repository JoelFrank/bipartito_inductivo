from torch import nn

class MlpPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.reset_parameters()
    def forward(self, x): return self.net(x)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): m.reset_parameters()
