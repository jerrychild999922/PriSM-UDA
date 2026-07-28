# models/segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkSegmentationNet(nn.Module):
    def __init__(self, latent_dim=512, output_size=256):
        """
        Model Architecture:
         - Flattens normalized landmarks (68x2).
         - Passes through three fully-connected layers to produce a latent feature, 
           then reshapes it to (latent_dim, 4, 4).
         - Decoder uses 6 ConvTranspose2d layers to progressively upsample to output_size x output_size.
         - Outputs logits of shape (B, 10, output_size, output_size).
        """
        super(LandmarkSegmentationNet, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(68 * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, latent_dim * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),         # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),           # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),           # 64x64 -> 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),            # 128x128 -> 256x256
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=3, padding=1)
        )
        
    def forward(self, landmarks):
        batch_size = landmarks.size(0)
        x = landmarks.reshape(batch_size, -1)  # (B, 136)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(batch_size, -1, 4, 4)  # (B, latent_dim, 4, 4)
        x = self.decoder(x)              # (B, 10, output_size, output_size)
        return x