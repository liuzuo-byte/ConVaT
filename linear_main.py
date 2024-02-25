import datetime
from operator import truediv
import operator
from functools import reduce
from sklearn.decomposition import PCA
import os
from matplotlib import pyplot as plt
from matplotlib.colors import  TABLEAU_COLORS,XKCD_COLORS
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import argparse
from tqdm import tqdm
from dataset import get_dataset, HyperX
from torch.utils import data
from utils import count_sliding_window, get_device, grouper, logger, metrics, sample_gt, sliding_window
from tools import AverageMeter
from VAE_model import Enc_VAE
from vit_copy import ViT_Encoder


parser = argparse.ArgumentParser(description="Run experiments on various hyperspectral datasets")

parser.add_argument('--dataset', type=str, default='IndianPines',
                    help="Choice one dataset for training"
                         "Dataset to train. Available:\n"
                         "PaviaU"
                         "Houston"
                         "IndianPines"
                         "KSC"
                         "Botswana"
                         "Salinas")
parser.add_argument('--folder', type=str, default='../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")
    
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
    
parser.add_argument('--train_gt', action='store_true',
                           help="Samples use of training")
    
parser.add_argument('--test_gt', action='store_true',
                        help="Samples use of testing")
parser.add_argument('--load_data', type=str, default=None,
                        help="Samples use of training")
parser.add_argument('--epoch', type=int,
                        help="Training epochs")
parser.add_argument('--patch_size', type=int,
                        help="Size of the spatial neighbourhood (optional, if "
                            "absent will be set by the model)")
parser.add_argument('--batch_size', type=int,
                         help="Batch size (optional, if absent will be set by the model")



args = parser.parse_args()

DATASET = args.dataset

# 生成日志
file_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger = logger('./logs/logs-' + file_date + DATASET +'.txt')
logger.info("---------------------------------------------------------------------")
logger.info("-----------------------------Next run log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")

FOLDER = args.folder
CUDA_DEVICE = get_device(logger,args.cuda)
PATCH_SIZE = args.patch_size
TRAIN_GT = args.train_gt
LOAD_DATA = args.load_data
TEST_GT = args.test_gt
EPOCH = args.epoch
BATCH_SIZE = args.batch_size

hyperparams = vars(args)

img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(logger, DATASET, FOLDER)


def applyPCA(X, numComponents=15):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True,svd_solver='full')#不加full会报数组错误
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

# Number of spectral bands
N_BANDS = img.shape[-1]
print(N_BANDS)

img = applyPCA(img,numComponents=200)#在main函数里降维

N_CLASSES = len(LABEL_VALUES)
# # Number of spectral bands
# N_BANDS = img.shape[-1]

hyperparams.update(
{'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)


if LOAD_DATA:
    train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
    test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
    train_gt = np.load(train_gt_file, 'r')
    logger.info("Load train_gt successfully!(PATH:{})".format(train_gt_file))
    logger.info("{} samples selected for training(over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    logger.info("Training Percentage:{:.2}".format(np.count_nonzero(train_gt)/np.count_nonzero(gt)))
    test_gt = np.load(test_gt_file, 'r')
    logger.info("Load train_gt successfully!(PATH:{})".format(test_gt_file))
    logger.info("{} samples selected for testing(over {})".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))


train_dataset = HyperX(img, train_gt, **hyperparams)
train_loader = data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    #pin_memory=hyperparams['device'],
                                    shuffle=True)

test_dataset = HyperX(img, test_gt, **hyperparams)
test_loader = data.DataLoader(test_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    #pin_memory=hyperparams['device'],
                                    shuffle=True)

# val_gt, test_dataset = sample_gt(gt, 0.05)#只用到了test_gt

# val_dataset = HyperX(img, val_gt, **hyperparams)
# val_loader = data.DataLoader(val_dataset,
#                                     #pin_memory=hyperparams['device'],
#                                     batch_size=hyperparams['batch_size'])


model = ViT_Encoder(image_size = 15,patch_size = 3,dim = 1024,depth = 2,heads = 8,
                        mlp_dim = 64,channels = 200,dropout = 0.1,emb_dropout = 0.1).cuda()

def L2_Norm(data):#估计是用来正则化特征
    norm=np.linalg.norm(data, ord=2)
    return truediv(data,norm)

model.load_state_dict(torch.load('/home/lmm/liuzuo/VAE_BYOL_HSI/checkpoints/VAE_BYOL_param/100%IP_1re+1con_lr1e-4'))
model = model.to(args.cuda)
classifier = nn.Linear(in_features=1024, out_features=17, bias=True).to(args.device)
classifier.to(args.cuda)


# #保存编码器提取的特征和标签
# Encoder_Features = []
# Label = []
# print('Start save patch features...')
# for i, (data,_,label) in enumerate(tqdm(train_loader)):
#     data=data.cuda().float()
#     model.eval()
#     feature,mu, sigma = model(data)
#     for num in range(len(feature)):
#         Encoder_Features.append(np.array(feature[num].cpu().detach().numpy()))#保存编码器特征到Encoder_Features[]
#         Label.append(np.array(label[num].cpu().detach().numpy()))#保存特征的标签到Label[]

# # Encoder_Features = L2_Norm(Encoder_Features)#961*1024（961个样本，每个样本特征128维）
# Encoder_Features = Encoder_Features#961*1024（961个样本，每个样本特征128维）
# Label = np.array(Label)#961*1


# #T-SNE可视化编码特征
# tsne = TSNE(n_components=2, random_state=33)
# X_tsne = tsne.fit_transform(Encoder_Features)   #X_tsne为降成2维后的特征
# plt.figure(figsize=(12, 12),dpi=800)    #dpi为像素

# colors =["#00F5FF", "#FF6A6A", "#7FFFD4", "#7FFF00", "#EEE685", "#FFC125", "#8B658B", "#0000FF", "#FF8247", "#CD2626", 
#          "#9370DB", "#1E90FF", "#FFBBFF", "#CAE1FF", "#00F5FF", "#800000", "#008B8B", "#00FFFF", "#D2B48C", "#FA8072"]

# for idx in range(1,N_CLASSES):
#     x_ = X_tsne[Label==idx]
#     plt.scatter(x_[:, 0], x_[:, 1],color = colors[idx],label=idx)
# plt.legend()#画出标签颜色
# plt.savefig('ConVaT_SA11'+'.png')
# plt.show()




# define optimizer
optimizer = torch.optim.Adam(classifier.parameters(), lr = 1e-3)#lr取0.001或0.01
# optimizer2 = torch.optim.SGD(classifier.parameters(), lr = 5e-3)#lr取0.001
# optimizer1 = torch.optim.SGD(model.parameters(), lr=3e-6,weight_decay=0.0005)#0.001较好

loss = torch.nn.CrossEntropyLoss()
loss_meter = AverageMeter(name='Loss')
acc_meter = AverageMeter(name='Accuracy')

# Start training
for e in tqdm(range(1, EPOCH + 1), desc="Training the network"):
    
    loss_meter.reset()
    model.eval()
    # model.train()
    classifier.train() 
        
    for batch_idx, (data,_,label) in enumerate(train_loader):
        
        data, label = data.to(args.cuda), label.to(args.cuda)
        classifier.zero_grad()
        

        with torch.no_grad():

            feature,_,_ = model.forward(data)

        pred = classifier.forward(feature)

        loss = F.cross_entropy(pred, label)

        loss.backward()
        optimizer.step()
        # optimizer2.step()
        loss_meter.update(loss.item())
                
classifier.eval()
model.eval()
correct, total = 0, 0
acc_meter.reset()
for idx, (images,_,labels) in enumerate(test_loader):
    
    with torch.no_grad():
        feature,_,_ = model(images.to(args.cuda))
        preds = classifier(feature).argmax(dim=1)
        correct = (preds == labels.to(args.cuda)).sum().item()
        acc_meter.update(correct/preds.shape[0])
print(f'Accuracy = {acc_meter.avg*100:.2f}')


def test(net,classifier,img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    # classifier.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = True
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    # probs = np.zeros(img.shape[:2] + (n_classes,))
    probs = np.zeros(img.shape[:2])
    img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'reflect')
    
    iterations = count_sliding_window(img, step=1, window_size=(patch_size, patch_size))
    
    for batch in tqdm(grouper(batch_size, sliding_window(img, step=1, window_size=(patch_size, patch_size))),
                      total=(iterations//batch_size),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            data = [b[0] for b in batch]

            data = np.copy(data)

            data = data.transpose(0, 3, 1, 2)

            data = torch.from_numpy(data)


            indices = [b[1:] for b in batch]

            data = data.to(device)
            data = data.type(torch.cuda.FloatTensor)
            # print(data.shape)
            output,_,_= net(data)
            # print(output.shape)
            output = classifier(output)#32*17
            _,output = torch.max(output, dim=1)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')
            if center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x, y] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

# --------------------------------------------------------------------------------------------
probabilities = test(model,classifier,img, hyperparams)#[145,145]
print("***********************************")
print(probabilities.shape)
results = metrics(probabilities, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
print(results)
        
logger.info('----------Training result----------')
logger.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
logger.info("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
logger.info("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
logger.info("\nAA:\n{}".format(np.mean(np.around(results['F1 scores'], 4)[1:])))
logger.info("\nKappa:\n{:.4f}".format(results['Kappa']))

