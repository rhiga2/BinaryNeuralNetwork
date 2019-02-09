python bitwise_mnist.py -exp relu_mnist -ba identity -e 64 -lr 1e-5 -dropout 0.2
python bitwise_mnist.py -exp nocw_nocg_mnist -ba ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp nocw_cg_mnist -ba clipped_ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp cw_cg_mnist -ba clipped_ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp cw_nocg_mnist -ba -ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
