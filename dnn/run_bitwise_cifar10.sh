# python bitwise_cifar10.py -exp resnet18 -e 250 -b 128 -d 3 -lr 1e-3 -lrd 0.5 -dp 100 -ib tanh -wb identity -bnm 0.01 -do 0.2
# python bitwise_cifar10.py -exp bresnet18 -e 250 -b 64 -d 3 -lr 1e-3 -lrd 0.5 -dp 100  -lf resnet18.model -ib clipped_ste -wb clipped_ste -bnm 0.2 -cw
python bitwise_cifar10.py -exp vgg16 -e 250 -b 128 -d 3 -lr 1e-3 -lrd 0.5 -dp 100 -ib tanh -wb identity -bnm 0.01 -do 0.2 -model vgg16
