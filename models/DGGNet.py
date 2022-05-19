#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from .dgcnn_group import DGCNN_Grouper
from models.build import MODELS
from utils import misc

class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
            nn.Conv1d(3,3,1),
            nn.Conv1d(3,3,1)
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2

class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x = torch.cat((torch.squeeze(x_128,dim=3), torch.squeeze(x_256,dim=3), torch.squeeze(x_512,dim=3)), dim=1)
        x = misc.fps(x.transpose(2, 1).contiguous(), 128).transpose(2, 1).contiguous()
        return x

class Latentfeature(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.maxpool = torch.nn.MaxPool1d(128, 1)
        self.grouper0 = DGCNN_Grouper()
        self.convlayer0 =Convlayer(point_scales = self.point_scales_list[0])
        self.grouper1 = DGCNN_Grouper()
        self.convlayer1 = Convlayer(point_scales=self.point_scales_list[1])
        self.grouper2 = DGCNN_Grouper()
        self.convlayer2 = Convlayer(point_scales=self.point_scales_list[2])
        self.conv1 = torch.nn.Conv1d(3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        coor0, f0 = self.grouper0(x[0].transpose(2,1).contiguous())
        out0 = self.convlayer0(x[0])
        out0 = self.maxpool(torch.cat((f0, out0), dim=1))
        coor1, f1 = self.grouper1(x[1].transpose(2,1).contiguous())
        out1 = self.convlayer1(x[1])
        out1 = self.maxpool(torch.cat((f1, out1), dim=1))
        coor2, f2 = self.grouper2(x[2].transpose(2,1).contiguous())
        out2 = self.convlayer2(x[2])
        out2 = self.maxpool(torch.cat((f2, out2), dim=1))

        latentfeature = torch.cat((out0, out1, out2), dim=2)
        latentfeature = latentfeature.transpose(1, 2).contiguous()
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature).contiguous()))
        latentfeature = torch.squeeze(latentfeature, 1)

        return latentfeature

@MODELS.register_module()
class DGGnet(nn.Module):
    def __init__(self, config, **kwargs):
        super(DGGnet, self).__init__()
        self.crop_point_num = config.crop_point_num
        self.latentfeature = Latentfeature(config.num_scales, config.each_scales_size, config.point_scales_list)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)
        self.fc3_1 = nn.Linear(256, 64 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 12, 1)

        self.fd1 = Fold(256, step=8, hidden_dim=64)
        self.fd2 = Fold(512, step=16, hidden_dim=128)
        self.fd3 = Fold(1024, step=24, hidden_dim=256)

        self.conv2 = torch.nn.Conv1d(256,256,1)
        self.conv3 = torch.nn.Conv1d(576,512,1)
        self.conv3_1 = torch.nn.Conv1d(512,512,1)

        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(576)

        self.fcCat1 = nn.Linear(6,3)
        self.fcCat2 = nn.Linear(6,3)
        self.fcCat3 = nn.Linear(6,3)

    def forward(self, x):
        x = self.latentfeature(x)
        x_2 = F.relu(self.fc2(x))
        x_3 = F.relu(self.fc3(x_2))

        fd1_xyz=self.fd1(x_3).transpose(2,1).contiguous()

        fd2_xyz=self.fd2(x_2).transpose(2,1).contiguous()
        fd1_xyz_expand = torch.unsqueeze(fd1_xyz, dim=2)
        fd2_xyz=fd2_xyz.reshape(-1,64,4,3)
        fd2_xyz=fd2_xyz+fd1_xyz_expand
        fd2_xyz=fd2_xyz.reshape(-1,256,3)

        fd3_xyz=self.fd3(x).transpose(2,1).contiguous()
        fd3_xyz=self.conv3(fd3_xyz)
        fd3_xyz=self.conv3_1(fd3_xyz)
        fd2_xyz_expand = torch.unsqueeze(fd2_xyz, dim=2)
        fd3_xyz=fd3_xyz.reshape(-1,256,2,3)
        fd3_xyz=fd3_xyz+fd2_xyz_expand
        fd3_xyz=fd3_xyz.reshape(-1,512,3)

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat)

        pc3_feat = F.relu(self.fc1_1(x))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 4, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 256, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 256, int(self.crop_point_num / 256), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)


        pc1_xyz=self.fcCat1(torch.cat((pc1_xyz,fd1_xyz),2))
        pc2_xyz=self.fcCat2(torch.cat((pc2_xyz,fd2_xyz),2))
        pc3_xyz=self.fcCat3(torch.cat((pc3_xyz,fd3_xyz),2))

        return pc1_xyz, pc2_xyz, pc3_xyz


