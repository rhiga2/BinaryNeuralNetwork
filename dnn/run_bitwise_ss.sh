python bitwise_ss.py -exp net_tanh -a identity -wb tanh -ib tanh -ub -lr 1e-4
python bitwise_ss.py -exp net_nocw_cg_tanh -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -lr 1e-5
python bitwise_ss.py -exp net_cw_cg_tanh -a identity -wa clipped_ste -ib clipped_ste -ub -lf net_tanh.model -cw -lr 1e-5
python bitwise_ss.py -exp net_cw_nocg_tanh -a identity -wa ste -ib ste -ub -lf net_tanh.model -cw -lr 1e-5
