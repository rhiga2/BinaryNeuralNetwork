python bitwise_cifar10.py -exp resnet18 -e 200 -b 128 -d 3 -lr 1e-4 -lrd 0.9 --pretrained -ib tanh -wb identity -bnm 0.01 -do 0.1
# python bitwise_cifar10.py -exp bresnet18 -e 100 -b 64 -d 3 -lr 1e-6 -lf resnet18.model -ba clipped_ste -cw
