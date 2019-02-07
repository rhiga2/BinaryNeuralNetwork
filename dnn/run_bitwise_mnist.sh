python bitwise_mnist.py -exp relu_mnist -wb identity -ib identity -e 150 -dropout 0.4 -lr 1e-4
python bitwise_mnist.py -exp nocw_nocg_mnist -wb ste -ib ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-4
python bitwise_mnist.py -exp nocw_cg_mnist -wb clipped_ste -ib clipped_ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-4
python bitwise_mnist.py -exp cw_cg_mnist -wb clipped_ste -ib clipped_ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-4
python bitwise_mnist.py -exp cw_nocg_mnist -wb ste -ib ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-4
