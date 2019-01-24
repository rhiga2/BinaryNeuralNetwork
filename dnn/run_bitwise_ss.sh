python bitwise_ss.py -exp net_tanh -a identity -wb tanh -ib tanh -ub -lr 1e-4 -e 256
python bitwise_ss.py -exp net_nocw_cg_tanh -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -lr 1e-5 -e 256 -dropout 0
python bitwise_ss.py -exp net_cw_cg_tanh -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -cw -lr 1e-5 -e 256 -dropout 0
python bitwise_ss.py -exp net_cw_nocg_tanh -a identity -wb ste -ib ste -ub -lf net_tanh.model -cw -lr 1e-5 -e 256 -dropout 0
python bitwise_ss.py -exp net_nocw_nocg_tanh -a identity -wb ste -ib ste -ub -lf net_tanh.model -lr 1e-5 -e 256 -dropout 0
