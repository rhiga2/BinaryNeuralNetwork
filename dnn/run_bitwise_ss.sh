python bitwise_ss.py -exp mlp_ss -ib tanh -wb ptanh -e 250 -b 32  -dropout 0.2 -bnm 0.01 -lr 1e-3 -lrd 0.01 -dp 150
python bitwise_ss.py -exp bmlp_ss -ib clipped_ste -wb ptanh_ste -lf mlp_ss.model -cw -e 150 -b 32 -bnm 0.2 -lrd 0.01 -dp 100 -lr 1e-3
# python bitwise_ss.py -exp cw_nocg_ss -ib ste -wb ste -lf tanh_ss.model -cw -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-4
# python bitwise_ss.py -exp nocw_nocg_ss -ib ste -wb ste -lf tanh_ss.model -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-4
# python bitwise_ss.py -exp nocw_cg_ss -ib clipped_ste -wb clipped_ste -lf tanh_ss.model -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-4
