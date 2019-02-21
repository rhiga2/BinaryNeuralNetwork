# python bitwise_cifar10.py -exp resnet18 -e 250 -b 128 -d 3 -lr 1e-3 -lrd 1.0 -dp 100 -ib tanh -wb identity -bnm 0.01 -do 0.3
python bitwise_cifar10.py -exp bresnet18 -e 100 -b 64 -d 3 -lr 1e-4 -lf resnet18.model -ib clipped_ste -wb clipped_ste -bnm 0.2 -cw
