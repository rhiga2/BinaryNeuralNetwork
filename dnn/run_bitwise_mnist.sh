python bitwise_ss.py -exp relu_mnist -wb identity -ib identity -ub -e 150 -dropout 0.3
python bitwise_ss.py -exp nocw_nocg_mnist -wb ste -ib ste -ub -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as
python bitwise_ss.py -exp nocw_cg_mnist -wb clipped_ste -ib clipped_ste -ub -lf relu_mnist.model -e 150 -bnm 0.2 -lrd 0.90 -as
python bitwise_ss.py -exp cw_cg_mnist -wb clipped_ste -ib clipped_ste -ub -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as
python bitwise_ss.py -exp cw_nocg_mnist -wb ste -ib ste -ub -lf relu_mnist.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as
