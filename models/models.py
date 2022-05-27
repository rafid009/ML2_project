from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfcn.models.fcn8s import FCN8s
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OccupancyEncoderCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OccupancyEncoderCNN, self).__init__()
        self.fcn8 = FCN8s()
        # self.fcn = FCN(in_channel, out_channel)
        state_dict = torch.load(FCN8s.download())

        self.fcn8.load_state_dict(state_dict)
        
        names = []
        for name, module in self.fcn8.named_modules():
            names.append(name)

        setattr(self.fcn8, names[1], nn.Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(100, 100)))
        modified_last_layers = [nn.Conv2d(4096, out_channel, kernel_size=(1, 1), stride=(1, 1)),
                                nn.Conv2d(256, out_channel, kernel_size=(1, 1), stride=(1, 1)),
                                nn.Conv2d(512, out_channel, kernel_size=(1, 1), stride=(1, 1)),
                                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), bias=False),
                                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(16, 16), stride=(8, 8), bias=False),
                                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), bias=False)]
        i = 0
        for name in names[-6:]:
            setattr(self.fcn8, name, modified_last_layers[i])
            i += 1
        self.fcn8 = self.fcn8.float()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fcn8(x))

class FCN(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding='same')
        # self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, out_channels, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o = self.conv1(x)
        # o = self.conv2(o)
        o = self.conv3(o)
        o = self.conv4(o)
        return self.sigmoid(o)

class DetectionEncoderCNN(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DetectionEncoderCNN, self).__init__()
        # self.fcn = FCN(in_channel, out_channel)
        self.conv3d = nn.Conv3d(in_channel, 1, 3, padding='same')
        self.fcn8 = FCN8s()
        state_dict = torch.load(FCN8s.download())

        self.fcn8.load_state_dict(state_dict)
        
        names = []
        for name, module in self.fcn8.named_modules():
            names.append(name)

        modified_last_layers = [nn.Conv2d(4096, out_channel, kernel_size=(1, 1), stride=(1, 1)),
                                nn.Conv2d(256, out_channel, kernel_size=(1, 1), stride=(1, 1)),
                                nn.Conv2d(512, out_channel, kernel_size=(1, 1), stride=(1, 1)),
                                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), bias=False),
                                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(16, 16), stride=(8, 8), bias=False),
                                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), bias=False)]
        i = 0
        for name in names[-6:]:
            setattr(self.fcn8, name, modified_last_layers[i])
            i += 1

        self.fcn8 = self.fcn8.float()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.fcn8(x))#self.fcn(x) 

class OccupancyDetectionModel(nn.Module):
    def __init__(self, occ_in_channel, detect_in_channel, out_channel):
        super(OccupancyDetectionModel, self).__init__()
        self.occ_features_encoder = OccupancyEncoderCNN(occ_in_channel, out_channel)
        self.detect_features_encoder = DetectionEncoderCNN(detect_in_channel, out_channel)
        self.conv2d = nn.Conv2d(out_channel, out_channel, 3, padding='same')
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, visit):
        occ = x['occupancy_feature'].to(device)
        detect = x[f'detection_feature_{visit}'].to(device)
        occ = self.occ_features_encoder(occ)
        detect = self.detect_features_encoder(detect)
        cat = detect * occ
        out = self.sigmoid(self.conv2d(cat))
        return torch.squeeze(out), occ, detect
