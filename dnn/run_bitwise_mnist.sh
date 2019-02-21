python bitwise_mnist.py -exp tanh_mnist -ib tanh -wb identity -e 250 -b 128 -lr 1e-2 -dropout 0 -bnm 0.01 -lrd 1.0 -dp 200
python bitwise_mnist.py -exp cw_cg_mnist -ib clipped_ste -wb clipped_ste -lf tanh_mnist.model -cw -e 100 -b 128 -bnm 0.2 -lrd 0.1 -dp 50 -lr 1e-3
python bitwise_mnist.py -exp cw_nocg_mnist -ib ste -wb ste -lf tanh_mnist.model -cw -e 100 -b 128 -bnm 0.2 -lrd 1 -dp 50 -lr 1e-3
python bitwise_mnist.py -exp nocw_nocg_mnist -ib ste -wb ste -lf tanh_mnist.model -e 100 -b 128 -bnm 0.2 -lrd 1 -lr -dp 50 1e-3
python bitwise_mnist.py -exp nocw_cg_mnist -ib clipped_ste -wb clipped_ste -lf tanh_mnist.model -e 100 -b 128 -bnm 0.2 -lrd 1 -dp 50 -lr 1e-3
