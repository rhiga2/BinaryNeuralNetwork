python bitwise_cifar10.py -exp resnet18 -e 250 -b 128 -d 3 -lr 1e-3 -lrd 0.1 -dp 200 -ib tanh -wb identity -bnm 0.01 -do 0.2
python bitwise_cifar10.py -exp bresnet18 -e 250 -b 64 -d 3 -lr 1e-3 -lrd 0.1 -dp 200  -lf resnet18.model -ib clipped_ste -wb clipped_ste -bnm 0.2 -cw
python bitwise_cifar10.py -exp resnet18_tanh -e 250 -b 128 -d 3 -lr 1e-3 -lrd 0.1 -dp 200 -ib tanh -wb tanh -bnm 0.01 -do 0.2
python bitwise_cifar10.py -exp bresnet18_tanh -e 250 -b 64 -d 3 -lr 1e-3 -lrd 0.1 -dp 200  -lf resnet18_tanh.model -ib clipped_ste -wb clipped_ste -bnm 0.2 -cw
python bitwise_cifar10.py -exp vgg16 -e 250 -b 128 -d 3 -lr 1e-3 -lrd 0.1 -dp 200 -ib tanh -wb identity -bnm 0.01 -do 0.2 -model vgg16
python bitwise_cifar10.py -exp bvgg16 -e 250 -b 64 -d 3 -lr 1e-3 -lrd 0.1 -dp 200  -lf vgg16.model -ib clipped_ste -wb clipped_ste -bnm 0.2 -cw -model vgg16
python bitwise_cifar10.py -exp vgg16_tanh -e 250 -b 128 -d 3 -lr 1e-3 -lrd 0.1 -dp 200 -ib tanh -wb tanh -bnm 0.01 -do 0.2 -model vgg16
python bitwise_cifar10.py -exp bvgg16_tanh -e 250 -b 64 -d 3 -lr 1e-3 -lrd 0.1 -dp 200  -lf vgg_tanh.model -ib clipped_ste -wb clipped_ste -bnm 0.2 -cw -model vgg16
