
python main.py --dataset 数据集名字  --patch_size 输入数据空间大小 --lr 学习率 --epoch 训练轮次 --run 重复进行多次实验，作用是取平均值 --load_data 加载数据比例共有（0.01，0.03，0.05，0.10，0.15）种  --class_balancing 类别平衡设置，无需更改


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

