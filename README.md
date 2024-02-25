ConVaT: A Variational Generative Transformer with Momentum Contrastive Learning for Hyperspectral Image Classification



#注意!!!!!!!：byol.py为论文中的MCL模块，vit_copy.py为论文中的VGT模块。

########运行线性分类器######：
IndianPines:
python linear_main.py --dataset IndianPines  --patch_size 15  --epoch 100 --batch_size 32 --load_data 0.10

PaviaU:
python linear_main.py --dataset PaviaU  --patch_size 15  --epoch 100 --batch_size 32 --load_data 0.10

Salinas:
python linear_main.py --dataset Salinas  --patch_size 15  --epoch 100 --batch_size 32 --load_data 0.05

LongKou:
python linear_main.py --dataset WHU-Hi-LongKou  --patch_size 15  --epoch 100 --batch_size 32 --load_data 0.05

HanChuan:
python linear_main.py --dataset WHU-Hi-HanChuan  --patch_size 15  --epoch 100 --batch_size 32 --load_data 0.05




######运行对比学习模型#######：
IndianPines:
python main.py --dataset IndianPines  --patch_size 15 --lr 0.0001 --epoch 100 --run 1 --load_data 0.30 --class_balancing --batch_size 64

PaviaU:
python main.py --dataset PaviaU  --patch_size 15 --lr 0.0001 --epoch 100 --run 1 --load_data 0.30 --class_balancing --batch_size 64

Salinas:
python main.py --dataset Salinas  --patch_size 15 --lr 0.0001 --epoch 100 --run 1 --load_data 0.30 --class_balancing --batch_size 64

LongKou:
python main.py --dataset WHU-Hi-LongKou  --patch_size 15 --lr 0.0001 --epoch 100 --run 1 --load_data 100 --class_balancing --batch_size 64

HanChuan:
python main.py --dataset WHU-Hi-HanChuan  --patch_size 15 --lr 0.0001 --epoch 100 --run 1 --load_data 0.30 --class_balancing --batch_size 64

#目前batch_size为64，head为8最佳性能

