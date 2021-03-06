python bitwise_cifar10.py -exp vgg16 -model vgg -e 200 -b 64 -d 0 -lr 5e-3 -lrd 0.1 -dp 150 -ib tanh -wb identity -do 0.2
python bitwise_cifar10.py -exp vgg16_tanh -model vgg -e 200 -b 64 -d 0 -lr 5e-3 -lrd 0.1 -dp 150 -ib tanh -wb tanh -do 0.2
python bitwise_cifar10.py -exp bvgg16_noinit -model vgg -e 200 -b 64 -d 0 -lr 5e-3 -lrd 0.1 -dp 150 -ib tanh_ste -wb tanh_ste -bnm 0.2 -cw
python bitwise_cifar10.py -exp bvgg16 -model vgg -e 200 -b 64 -d 0 -lr 5e-3 -lrd 0.1 -dp 150 -lf vgg16.model -ib tanh_ste -wb tanh_ste -bnm 0.2 -cw
python bitwise_cifar10.py -exp bvgg16_tanh -model vgg -e 200 -b 64 -d 0 -lr 5e-3 -lrd 0.1 -dp 150 -lf vgg16_tanh.model -ib tanh_ste -wb tanh_ste -bnm 0.2 -cw

# python bitwise_cifar10.py -exp bvgg16_25 -ib tanh_ste -wb tanh_ste -e 300 -lf vgg16.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -sparsity 25
# python bitwise_cifar10.py -exp bvgg16_33 -ib tanh_ste -wb tanh_ste -e 300 -lf vgg16.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -sparsity 33
# python bitwise_cifar10.py -exp bvgg16_50 -ib tanh_ste -wb tanh_ste -e 300 -lf vgg16.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -sparsity 50
# python bitwise_cifar10.py -exp vgg16_ug -ib tanh -wb identity -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100 -ug
# python bitwise_cifar10.py -exp bvgg16_ug -ib tanh_ste -wb tanh_ste -e 300 -lf vgg16_ug.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -ug
