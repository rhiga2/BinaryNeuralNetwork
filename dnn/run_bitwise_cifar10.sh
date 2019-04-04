# python bitwise_cifar10.py -exp vgg16 -model vgg16 -e 300 -b 128 -d 3 -lr 1e-3 -lrd 0.8 -dp 50 -ib tanh -wb identity -bnm 0.01 -do 0.4
python bitwise_cifar10.py -exp vgg16_tanh -model vgg16 -e 300 -b 128 -d 0 -lr 1e-3 -lrd 0.8 -dp 50 -ib tanh -wb tanh -bnm 0.01 -do 0.4
# python bitwise_cifar10.py -exp bvgg16_noinit -model vgg16 -e 300 -b 128 -d 3 -lr 1e-3 -lrd 0.3 -dp 100 -ib tanh_ste -wb tanh_ste -bnm 0.2 -cw
# python bitwise_cifar10.py -exp bvgg16 -model vgg16 -e 300 -b 128 -d 3 -lr 1e-3 -lrd 0.3 -dp 100 -lf vgg16.model -ib tanh_ste -wb tanh_ste -bnm 0.2 -cw
python bitwise_cifar10.py -exp bvgg16_tanh -model vgg16 -e 300 -b 128 -d 0 -lr 1e-3 -lrd 0.3 -dp 100 -lf vgg16_tanh.model -ib tanh_ste -wb tanh_ste -bnm 0.2 -cw
