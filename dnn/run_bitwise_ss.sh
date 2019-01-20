python bitwise_ss.py -exp net_relu -a relu -wb tanh -ib tanh -ub -lr 1e-4
python bitwise_ss.py -exp net_nocw_cg_relu -a relu -wb clipped_ste -ib clipped_ste -ub -lf net_relu.model -lr 1e-4
python bitwise_ss.py -exp net_cw_cg_relu -a relu -wa clipped_ste -ib clipped_ste -ub -lf net_relu.model -cw -lr 1e-4
python bitwise_ss.py -exp net_cw_nocg_relu -a relu -wa ste -ib ste -ub -le net_relu.model -cw -lr 1e-4
