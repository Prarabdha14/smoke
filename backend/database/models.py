import torch
import torch.nn as nn
from database.db import db

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ImageRecord(db.Model):
    """Database model for storing image analysis results"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f"<ImageRecord {self.filename} - {self.prediction}>"


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),  # model.0
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # model.1
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Corrected layer - changed from 128->128 to 128->256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # model.4
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Residual blocks
            *[ResidualBlock(256) for _ in range(n_residual_blocks)],
            
            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)