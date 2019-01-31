python bitwise_ss.py -exp tanh -a identity -wb identity -ib tanh -ub -lr 1e-3 -e 128
python bitwise_ss.py -exp nocw_nocg -a identity -wb ste -ib ste -ub -lf net_tanh.model -lr 1e-4 -e 128 -dropout 0
python bitwise_ss.py -exp nocw_cg -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -lr 1e-4 -e 128 -dropout 0
python bitwise_ss.py -exp cw_cg -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -cw -lr 1e-4 -e 128 -dropout 0
python bitwise_ss.py -exp cw_nocg -a identity -wb ste -ib ste -ub -lf net_tanh.model -cw -lr 1e-4 -e 128 -dropout 0
