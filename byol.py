import copy
import random
from einops import rearrange
import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import transforms 
from math import pi, cos 
from collections import OrderedDict
from arguments import argument1,argument2, argument3,channel_shuffle,radiation_noise,GridMask
from VAE_model import Enc_VAE,Dec_VAE
from vit_copy import ViT_Encoder,ViT_Decoder


grid_mask = GridMask()

# #这里使用的是3D-2D卷积做解码编码器器，我只需将这里换成VIT_Encoder,VIT_Decoder
# Enc_patch = Enc_VAE(channel=30,patch_size=25,output_dim=128).cuda()
# Dec_patch = Dec_VAE(channel=30,patch_size=25,input_dim=128).cuda()

Enc_patch = ViT_Encoder(image_size = 15,patch_size = 3,dim = 1024,depth = 2,heads = 8,
                        mlp_dim = 64,channels = 200,dropout = 0.1,emb_dropout = 0.1).cuda()

Dec_patch = ViT_Decoder(image_size = 15,patch_size = 3,dim = 1024,depth = 1,heads = 8,
                        mlp_dim = 64,channels = 200,dropout = 0.1,emb_dropout = 0.1).cuda()



#均方误差
criterion = torch.nn.MSELoss()

#L2损失函数
def D(x, y, version='simplified'):
    
    if version == 'original':
        y = y.detach()
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return (2 - 2 * (x * y).sum(dim=-1)).mean()
    elif version == 'simplified':
        return (2 - 2 * F.cosine_similarity(x,y.detach(), dim=-1)).mean()
    else:
        raise NotImplementedError



#infonce损失函数
def NT_XentLoss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

#KL损失函数
def KL(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


class MLP(nn.Module):
    def __init__(self, dim=1024,hidden_size = 2048,projection_size=1024):#hidden_size = 2048
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.layer(x)



class V_BYOL(nn.Module):
    def __init__(self):
        super().__init__()

        self.online_encoder = Enc_patch
        self.online_decoder = Dec_patch

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_MLP = MLP()
        self.online_MLP = MLP()



    #target网络参数动量更新
    @torch.no_grad()
    def update_moving_average(self,t=0.99):#目前模型来说，T=0.99模型为最佳
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = t * target.data + (1 - t) * online.data

            
    def forward(self, xx1,xx2):
        x1 = xx1
        x2 = xx2
        #x.shape=(128,30,25,25)
        #在这里分别对每个图片进行数据增强,如果在HyperX()进行增强，造成所有数据都是进行的一样的增强！！！
        # x1=x
        # x2=x
        for i in range(int(xx1.shape[0])):
            patch1 = xx1[i, :, :, :]
            X_aug1 = argument1(patch1)

            patch2 = xx2[i,:, :, :]
            X_aug2 = argument2(patch2)
            x1[i, :, :, :] = X_aug1
            x2[i,:, :, :] = X_aug2

        

        # x = x.unsqueeze(0) #x.shape(1,128,30,25,25)
        # x = x.permute(1,0,2,3,4)

        # x1 = x1.unsqueeze(0) #x1.shape(1,128,30,25,25)
        # x1 = x1.permute(1,0,2,3,4)#x1.shape(128,1,30,25,25)，用于3D卷积运算

        # x2 = x2.unsqueeze(0)
        # x2 = x2.permute(1,0,2,3,4)


        target = rearrange(xx1,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 3, p2 = 3)#将数据展平，便于与decoder做损失
            

        E1,D1,P1   =  self.online_encoder,self.online_decoder,self.online_MLP
        E2,P2      =  self.target_encoder,self.target_MLP

        h, mu, sigma = E1(x1)
        h = P1(h)#给特征加mlp

        mu = mu.repeat(1,25)
        mu = mu.view(64,25,1024)

        sigma = sigma.repeat(1,25)
        sigma = sigma.view(64,25,1024)

        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()#在Vit_encoder里面学方差

        code=mu + sigma * torch.tensor(std_z, requires_grad=False).cuda()

        recon = D1(code)#用z1图像重建

        with torch.no_grad():

            z2,_,_ = E2(x2)
        p2 = P2(z2)

        

        L = criterion(target,recon) + KL(mu,sigma) + (NT_XentLoss(h,p2))
        # L = criterion(target,recon) + D(h,p2) + KL(mu,sigma)


        return L