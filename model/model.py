from torch import nn

class ChessEval(nn.Module):
    def __init__(self):
        super(ChessEval, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Flatten(),
            
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(True),
            
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x): return self.seq(x)
