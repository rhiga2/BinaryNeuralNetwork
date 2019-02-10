python bitwise_cifar10.py -exp resnet18 -e 100 -b 16 -d 3 -lr 1e-5 -pretrained -ba identity
python bitwise_cifar10.py -exp bresnet18 -e 100 -b 16 -d 3 -lr 1e-6 -lr resnet18.model -as -ba clipped_ste -cw
