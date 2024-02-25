import random
import time
import torch
from torch.utils import data
from sklearn.decomposition import PCA
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
from byol import V_BYOL
from dataset import get_dataset, HyperX

from utils import  get_device, sample_gt, compute_imf_weights, metrics, logger, display_dataset, display_goundtruth, \
    pca_dr
import argparse
import numpy as np
import warnings
import datetime
import visdom

# 忽略警告
warnings.filterwarnings("ignore")


# 配置项目参数
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
# parser.add_argument('--model', type=str, default='BYOL',
#                     help="Model to train.")

parser.add_argument('--folder', type=str, default='../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--run', type=int, default=1,
                    help="Running times.")

group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--sampling_mode', type=str, default='fixed',
                           help="Sampling mode (random sampling or disjoint, default:  fixed)")
group_dataset.add_argument('--training_percentage', type=float, default=0.05,
                           help="Percentage of samples to use for training")
group_dataset.add_argument('--validation_percentage', type=float,default=0.1,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")
group_dataset.add_argument('--train_gt', action='store_true',
                           help="Samples use of training")
group_dataset.add_argument('--test_gt', action='store_true',
                           help="Samples use of testing")
group_dataset.add_argument('--load_data', type=str, default=None,
                           help="Samples use of training")
group_dataset.add_argument('--sample_nums', type=int, default=20,
                           help="Number of samples to use for training and validation")           
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int,
                         help="Training epochs")
group_train.add_argument('--save_epoch', type=int, default=20,
                         help="Training save epoch")
group_train.add_argument('--patch_size', type=int,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")

# Data augmentation parameters
group_data = parser.add_argument_group('Data augmentation')

args = parser.parse_args()

RUN = args.run


# Dataset name
DATASET = args.dataset
# 生成日志
file_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger = logger('./logs/logs-' + file_date + DATASET +'.txt')
logger.info("---------------------------------------------------------------------")
logger.info("-----------------------------Next run log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")
# CUDA_DEVICE
CUDA_DEVICE = get_device(logger, args.cuda)
# Model name

# Target folder to store/download/load the datasets
FOLDER = args.folder
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Automated class balancing
SAMPLE_NUMS = args.sample_nums
LOAD_DATA = args.load_data
PATCH_SIZE = args.patch_size
CLASS_BALANCING = args.class_balancing
TRAINING_PERCENTAGE = args.training_percentage
TEST_STRIDE = args.test_stride
TRAIN_GT = args.train_gt
TEST_GT = args.test_gt
EPOCH = args.epoch
SAVE_EPOCH = args.save_epoch
hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(logger, DATASET, FOLDER)
print("$$$$$$$$$$$$$$$$$$$$$$$$$")
print(img.shape)

    
def applyPCA(X, numComponents=15):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True,svd_solver='full')#不加full会报数组错误
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    
img = applyPCA(img,numComponents=200)#在main函数里降维

# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of spectral bands
N_BANDS = img.shape[-1]
# Instantiate the experiment based on predefined networks
hyperparams.update(
    {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)



for i in range(RUN):
    
    if LOAD_DATA:
        train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
        test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
        print(train_gt_file)
        train_gt = np.load(train_gt_file, 'r')
        logger.info("Load train_gt successfully!(PATH:{})".format(train_gt_file))
        logger.info("{} samples selected for training(over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        logger.info("Training Percentage:{:.2}".format(np.count_nonzero(train_gt)/np.count_nonzero(gt)))
        test_gt = np.load(test_gt_file, 'r')
        logger.info("Load train_gt successfully!(PATH:{})".format(test_gt_file))
        logger.info("{} samples selected for training(over {})".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
    else:
        train_gt_file = '../dataset/' + DATASET + '/20/' + '/train_gt.npy'
        test_gt_file  = '../dataset/' + DATASET + '/20/' + '/test_gt.npy'
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode=SAMPLING_MODE)
        np.save(train_gt_file, train_gt)
        np.save(test_gt_file, test_gt)

        logger.info("Save train_gt successfully!(PATH:{})".format(train_gt_file))
        logger.info("Save test_gt successfully!(PATH:{})".format(test_gt_file))
    # logger.info("Running an experiment with the {} model, RUN [{}/{}]".format(MODEL, i + 1, RUN))
    logger.info("RUN:{}".format(i))
    mask = np.unique(train_gt)
    tmp = []
    for v in mask:
        tmp.append(np.sum(train_gt==v))
    print("类别：{}".format(mask))
    print("训练集大小{}".format(tmp))
    mask = np.unique(test_gt)
    tmp = []
    for v in mask:
        tmp.append(np.sum(test_gt==v))
    # print(mask)
    print("测试集大小{}".format(tmp))
    logger.info("{} samples selected for training(over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    logger.info("{} samples selected for testing(over {})".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
    # logger.info("Running an experiment with the {} model, RUN [{}/{}]".format(MODEL, i + 1, RUN))
    logger.info("RUN:{}".format(i))
    
    # val_gt, test_dataset = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)

    # Class balancing
    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        #    hyperparams.update({'weights': torch.from_numpy(weights)})
        hyperparams['weights'] = torch.from_numpy(weights).float().cuda()
    

    train_dataset = HyperX(img, train_gt, **hyperparams)
    
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   shuffle=True,
                                   drop_last=True)
    logger.info("Train dataloader:{}".format(len(train_loader)))

    # val_dataset = HyperX(img, val_gt, **hyperparams)
    # val_loader = data.DataLoader(val_dataset,
    #                              batch_size=hyperparams['batch_size'],
    #                              drop_last=True)
    # logger.info("Validation dataloader:{}".format(len(val_loader)))
    # test_dataset = HyperX(img, test_dataset, **hyperparams)
    # test_loader = data.DataLoader(val_dataset,
    #                              batch_size=hyperparams['batch_size'])
    logger.info('----------Training parameters----------')
for k,v in hyperparams.items():
	logger.info("{}:{}".format(k,v))




logger.info('---------- pretrain model training----------')

model = V_BYOL()
model.to(args.cuda)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,weight_decay=0.0005)#使用SGD可以训练到100epoch
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4,weight_decay=0.0005)#IP用1e-4,UP与SA用3e-5


#训练开始时间
t0 = time.time()
print('$$$$$t0$$$$$$$',t0)

for e in tqdm(range(1, EPOCH+1), desc='Training the network'):
    model.train()
    losses = np.zeros(1000000)
    avg_loss = 0.
    
    for batch_idx, (data1,data2,_) in enumerate(train_loader):
        data1 = data1.type(torch.cuda.FloatTensor)
        data2 = data2.type(torch.cuda.FloatTensor)
        loss = model.forward(data1,data2)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        model.update_moving_average()

        del(data1,data2,loss)
    avg_loss /= len(train_loader)
    
    # 在控制台打印信息
    tqdm.write(f"Epoch [{e}/{EPOCH}]    avg_loss:{avg_loss:.6f}")
    # 在日志中打印信息
    logger.debug(f"Epoch [{e}/{EPOCH}]    avg_loss:{avg_loss:.6f}")
    

#训练结束时间
t1 = time.time()
print('$$$$$t1$$$$$$$',t1)

spend1 = t1 - t0
print("time()方法用时：{}s".format(spend1))

model_path = '/home/lmm/liuzuo/VAE_BYOL_HSI/checkpoints/VAE_BYOL_param/' + '100%IP_1re+1con_lr1e-4'
torch.save(model.online_encoder.state_dict(), model_path)

logger.info('The pretrain model vit training successfully!!!')

print(f"Model saved to {model_path}")

log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger.info("-----------------------------Next run finish log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")