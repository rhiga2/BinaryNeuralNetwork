python bitwise_mnist.py -exp relu_mnist -ba identity -e 250 -b 128 -lr 1e-4 -dropout 0.2 -bnm 0.01 
python bitwise_mnist.py -exp cw_cg_mnist -ba clipped_ste -lf relu_mnist.model -cw -e 100 -b 128 -bnm 0.2 -lrd 0.50 -lr 1e-5 -as
python bitwise_mnist.py -exp cw_nocg_mnist -ba ste -lf relu_mnist.model -cw -e 100 -b 128 -bnm 0.2 -lrd 0.50 -lr 1e-5 -as
python bitwise_mnist.py -exp nocw_nocg_mnist -ba ste -lf relu_mnist.model -e 100 -b 128 -bnm 0.2 -lrd 0.50 -lr 1e-5 -as
python bitwise_mnist.py -exp nocw_cg_mnist -ba clipped_ste -lf relu_mnist.model -e 100 -b 128 -bnm 0.2 -lrd 0.50 -lr 1e-5 -as
