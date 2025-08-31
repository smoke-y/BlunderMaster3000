import torch
from torch import nn

class ChessEval(nn.Module):
    def __init__(self):
        super(ChessEval, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Input: 12x8x8
        # After block 1 & 2 (padding=1): 512x8x8
        # After block 3 (no padding): 512x6x6 (8-2=6)
        # After block 4 (no padding): 512x4x4 (6-2=4)
        conv_output_size = 512 * 4 * 4
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.value_head = nn.Linear(256, 1)
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        value = torch.tanh(self.value_head(x))
        return value
