import random
import time
import numpy as np
import torch

from torchvision.transforms import RandomResizedCrop
import torch
import random
import numpy as np
import math
from torch.distributions.beta import Beta
import torchvision.transforms.functional as F

def argument1(x):#空间随机像素擦除，中心点除外
    num = random.randint(0,7)
    if num >= 0 and num < 4:
        aug = np.fliplr(x)
        aug=aug.transpose(1,2,0)
        np.random.seed(int(time.time()))
        #aug = cv2.GaussianBlur(np.float32(aug), (7, 7), wz)
        noise=np.random.randint(0,2,(aug.shape[0],aug.shape[1]))
        noise[aug.shape[0]//2,aug.shape[1]//2]=1#确保中心没有噪声
        noise=noise[:,:,np.newaxis]
        noise=np.concatenate([noise]*aug.shape[2],2)
        aug=aug*noise
        aug=aug.transpose(2,0,1)
        Aug = torch.from_numpy(aug.copy())

    elif num > 3 and num < 8:
        aug = np.flipud(x)
        aug=aug.transpose(1,2,0)
        np.random.seed(int(time.time()))
        #aug = cv2.GaussianBlur(np.float32(aug), (7, 7), wz)
        noise=np.random.randint(0,2,(aug.shape[0],aug.shape[1]))
        noise[aug.shape[0]//2,aug.shape[1]//2]=1#确保中心没有噪声
        noise=noise[:,:,np.newaxis]
        noise=np.concatenate([noise]*aug.shape[2],2)
        aug=aug*noise
        aug=aug.transpose(2,0,1)
        Aug = torch.from_numpy(aug.copy())
    else:
        Aug = x
    return Aug

def argument2(x):#空间随机像素块擦除，中心点除外
    num = random.randint(0,7)
    if num >= 0 and num < 4:
        #print(x.shape)
        aug = np.fliplr(x)
        aug=aug.transpose(1,2,0)
        np.random.seed(int(time.time()))
        #aug = cv2.GaussianBlur(np.float32(aug), (7, 7), wz)
        noise=np.ones((aug.shape[0],aug.shape[1]))
        ran1=np.random.randint(0,aug.shape[0]-1)
        ran2=np.random.randint(ran1+1,aug.shape[0])
        ran3=np.random.randint(0,aug.shape[0]-1)
        ran4=np.random.randint(ran3+1,aug.shape[0])
        noise[ran1:ran2,ran3:ran4]=0
        noise[aug.shape[0]//2,aug.shape[1]//2]=1#确保中心没有噪声
        noise=noise[:,:,np.newaxis]
        noise=np.concatenate([noise]*aug.shape[2],2)
        aug=aug*noise
        aug=aug.transpose(2,0,1)
        Aug = torch.from_numpy(aug.copy())
    elif num >3 and num < 8:
        #print(x.shape)
        aug = np.flipud(x)
        aug=aug.transpose(1,2,0)
        np.random.seed(int(time.time()))
        #aug = cv2.GaussianBlur(np.float32(aug), (7, 7), wz)
        noise=np.ones((aug.shape[0],aug.shape[1]))
        ran1=np.random.randint(0,aug.shape[0]-1)
        ran2=np.random.randint(ran1+1,aug.shape[0])
        ran3=np.random.randint(0,aug.shape[0]-1)
        ran4=np.random.randint(ran3+1,aug.shape[0])
        noise[ran1:ran2,ran3:ran4]=0
        noise[aug.shape[0]//2,aug.shape[1]//2]=1#确保中心没有噪声
        noise=noise[:,:,np.newaxis]
        noise=np.concatenate([noise]*aug.shape[2],2)
        aug=aug*noise
        aug=aug.transpose(2,0,1)
        Aug = torch.from_numpy(aug.copy())
    return Aug

def argument3(data):#，3D空间随机掩码,ratio为掩码率
        a = torch.rand_like(data)
        zero = torch.zeros_like(data)
        one = torch.ones_like(data)
        b = torch.where(a > 0.3, one, zero) 
        return data*b


def channel_shuffle(x, groups):#通道洗牌

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x



class ContrastiveCrop(RandomResizedCrop):  #对比裁剪方法 # adaptive beta
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        # a == b == 1.0 is uniform distribution
        self.beta = Beta(alpha, alpha)

    def get_params(self, img, box, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        # width, height = F._get_image_size(img)
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = box
                ch0 = min(max(int(height * h0) - h//2, 0), height - h)
                ch1 = min(max(int(height * h1) - h//2, 0), height - h)
                cw0 = min(max(int(width * w0) - w//2, 0), width - w)
                cw1 = min(max(int(width * w1) - w//2, 0), width - w)

                i = ch0 + int((ch1 - ch0) * self.beta.sample())
                j = cw0 + int((cw1 - cw0) * self.beta.sample())
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, box):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, box, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

def radiation_noise(data, alpha_range=(0.6, 1.4), beta=1/5):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise



import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math
 
 
class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob
 
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)
 
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
 
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))
 
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
 
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)
 
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
 
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
 
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
 
        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
 
        mask = mask.expand_as(img)
        img = img * mask
 
        return img
 
 
class GridMask(nn.Module):
    def __init__(self, d1=96, d2=224, rotate=360, ratio=0.4, mode=1, prob=0.8):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)
 
    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)
 
    def forward(self, x):
        if not self.training:
            return x
 
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
 
        y = torch.cat(y).view(n, c, h, w)
 
        return y
