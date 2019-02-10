<<<<<<< HEAD
python bitwise_mnist.py -exp relu_mnist -ba identity -e 64 -lr 1e-5 -dropout 0.2
python bitwise_mnist.py -exp nocw_nocg_mnist -ba ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp nocw_cg_mnist -ba clipped_ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp cw_cg_mnist -ba clipped_ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp cw_nocg_mnist -ba ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
||||||| merged common ancestors
python bitwise_mnist.py -exp relu_mnist -ba identity -e 64 -lr 1e-5 -dropout 0.2
python bitwise_mnist.py -exp nocw_nocg_mnist -ba ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp nocw_cg_mnist -ba clipped_ste -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp cw_cg_mnist -ba clipped_ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
python bitwise_mnist.py -exp cw_nocg_mnist -ba -ste -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as -lr 1e-5
=======
python bitwise_mnist.py -exp relu_mnist -ba identity -e 100 -b 64 -lr 1e-5 -dropout 0.3
python bitwise_mnist.py -exp cw_cg_mnist -ba clipped_ste -lf relu_mnist.model -cw -e 100 -b 64 -bnm 0.2 -lrd 0.90 -lr 1e-6 -as
python bitwise_mnist.py -exp cw_nocg_mnist -ba ste -lf relu_mnist.model -cw -e 100 -b 64 -bnm 0.2 -lrd 0.90 -lr 1e-6 -as
python bitwise_mnist.py -exp nocw_nocg_mnist -ba ste -lf relu_mnist.model -e 100 -b 64 -bnm 0.2 -lrd 0.90 -lr 1e-6 -as
python bitwise_mnist.py -exp nocw_cg_mnist -ba clipped_ste -lf relu_mnist.model -e 100 -b 64 -bnm 0.2 -lrd 0.90 -lr 1e-6 -as
>>>>>>> dd2ad440c2d54c659de8cfcf49e35c648a226195
