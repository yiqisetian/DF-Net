#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))#x:[18,1,2048,3]->[18,64,2048,1]
        x = F.relu(self.bn2(self.conv2(x)))#x:[18,64,2048,1]->[18,64,2048,1]
        x_128 = F.relu(self.bn3(self.conv3(x))) #x:[18,64,2048,1]->x_128[18,128,2048,1]
        x_256 = F.relu(self.bn4(self.conv4(x_128)))#x_128[18,128,2048,1]->x_256[18,256,2048,1]
        x_512 = F.relu(self.bn5(self.conv5(x_256)))#x_256[18,256,2048,1]->x_512[18,512,2048,1]
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))#x_512[18,512,2048,1]->x_1024[18,1024,2048,1]
        x_128 = torch.squeeze(self.maxpool(x_128),2)#x_128[18,128,2048,1]->x_128[18,128,1]
        x_256 = torch.squeeze(self.maxpool(x_256),2)#x_256[18,256,2048,1]->x_256[18,256,1]
        x_512 = torch.squeeze(self.maxpool(x_512),2)#x_512[18,512,2048,1]->x_512[18,512,1]
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)#x_1024[18,1024,2048,1]->x_1024[18,1024,1]
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x

class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):#[[24,2048,3],[24,1024,3],[24,512,3]]
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))  #这个模块无论进去的维度是多少，出来都是[24,1920,1]
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        #上面就算把这三种分辨率的统一到一样的大小
        latentfeature = torch.cat(outs,2)  #[24,1920,3]
        latentfeature = latentfeature.transpose(1,2) #[24,3,1920]
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))#[24,1,1920]
        latentfeature = torch.squeeze(latentfeature,1) #[24,1920]
#        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))  
#        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
#        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
#        latentfeature = latentfeature + latentfeature_64
#        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
#        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
#        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
#        latentfeature = latentfeature + latentfeature_256
#        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
#        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
#        latentfeature = self.maxpool(latentfeature)
#        latentfeature = torch.squeeze(latentfeature,2)
        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class _netG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,6,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        
    def forward(self,x):#[[24,2048,3],[24,1024,3],[24,512,3]]
        x = self.latentfeature(x)  #[24,1920]
        x_1 = F.relu(self.fc1(x)) #x:[24,1920]->x_1:[24,1024]
        x_2 = F.relu(self.fc2(x_1)) #x:[24,1024]->x_2:[24,512]
        x_3 = F.relu(self.fc3(x_2))  #x:[24,512]->x_3:[24,256]
        
        
        pc1_feat = self.fc3_1(x_3)#x_3:[24,256]->px1_feat:[24,192]
        pc1_xyz = pc1_feat.reshape(-1,64,3) #64x3 center1 pc1_feat:[24,192]->px1_xyz[24,64,3]
        
        pc2_feat = F.relu(self.fc2_1(x_2))#x_2:[24,512]->pc2_feat[24,8192]
        pc2_feat = pc2_feat.reshape(-1,128,64)#pc2_feat[24,8192]->[24,128,64]
        pc2_xyz =self.conv2_1(pc2_feat) #pc2_feat[24,128,64]->pc2_xyz[24,6,64]
        
        pc3_feat = F.relu(self.fc1_1(x_1))  #x_1[1024]->pc3_feat[24,65536]
        pc3_feat = pc3_feat.reshape(-1,512,128) #pc3_feat[24,65536]->[24,512,128]
        pc3_feat = F.relu(self.conv1_1(pc3_feat)) #[24,512,128]->[24,512,128]
        pc3_feat = F.relu(self.conv1_2(pc3_feat))#[24,512,128]-[24,256,128]
        pc3_xyz = self.conv1_3(pc3_feat) #12x128 fine pc3_feat[24,256,128]->pc3_xyz[24,12,128]
        
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)#pc1_xyz:[24,64,3]->pc1_xyz_expand:[24,64,1,3]
        pc2_xyz = pc2_xyz.transpose(1,2)#[24,6,64]->[24,64,6]
        pc2_xyz = pc2_xyz.reshape(-1,64,2,3) #[24,64,2,3]
        pc2_xyz = pc1_xyz_expand+pc2_xyz #[24,64,1,3]+[24,64,2,3]=[24,64,2,3]
        pc2_xyz = pc2_xyz.reshape(-1,128,3) #[24,128,3]
        
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2) #pc2_xyz[24,128,3]->px2_xyz_expand[24,128,1,3]
        pc3_xyz = pc3_xyz.transpose(1,2)#[24,128,12]
        pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3) #[24,128,4,3]
        pc3_xyz = pc2_xyz_expand+pc3_xyz#[24,128,4,3]
        pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3)#[24,512,3]
        
        return pc1_xyz,pc2_xyz,pc3_xyz #center1:px1_xyz[24,64,3] ,center2:pc2_xyz[24,128,3] ,fine:pc3_xyz[24,512,3]

class _netlocalD(nn.Module):
    def __init__(self,crop_point_num):
        super(_netlocalD,self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):#x:[24,1,512,3]
        x = F.relu(self.bn1(self.conv1(x)))#x:[24,1,512,3]->[24,64,512,1]把最后一个维度也给合并了？
        x_64 = F.relu(self.bn2(self.conv2(x)))#x:[24,1,512,3]->x_64:[24,64,512,1]
        x_128 = F.relu(self.bn3(self.conv3(x_64)))#x_64:[24,64,512,1]->x_128[24,128,512,1]
        x_256 = F.relu(self.bn4(self.conv4(x_128)))#x_128[24,128,512,1]->x_256[24,256,512,1]
        x_64 = torch.squeeze(self.maxpool(x_64))  #[24,64,512,1]->[24,64]
        x_128 = torch.squeeze(self.maxpool(x_128)) #[24,128,512,1]->[24,128]
        x_256 = torch.squeeze(self.maxpool(x_256))#[24,256,512,1]->[24,256]
        Layers = [x_256,x_128,x_64]
        x = torch.cat(Layers,1)  #为什么不直接cat？这人编程不太行[24,448]
        x = F.relu(self.bn_1(self.fc1(x))) #[24,448]->[24,256]
        x = F.relu(self.bn_2(self.fc2(x)))#[24,256]->[24,128]
        x = F.relu(self.bn_3(self.fc3(x)))#[24,128]->[24,16]
        x = self.fc4(x)#[24,1]
        return x

if __name__=='__main__':
    input1 = torch.randn(64,2048,3)
    input2 = torch.randn(64,512,3)
    input3 = torch.randn(64,256,3)
    input_ = [input1,input2,input3]
    netG=_netG(3,1,[2048,512,256],1024)
    output = netG(input_)
    print(output)
