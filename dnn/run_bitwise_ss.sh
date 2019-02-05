python bitwise_ss.py -exp tanh -a identity -wb identity -ib tanh -ub -e 150 -dropout 0.3
python bitwise_ss.py -exp nocw_nocg -a identity -wb ste -ib ste -ub -lf net_tanh.model -e 150 -bnm 0.2 -lrd 0.90
python bitwise_ss.py -exp nocw_cg -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -e 150 -bnm 0.2 -lrd 0.90
python bitwise_ss.py -exp cw_cg -a identity -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -cw -e 150 -bnm 0.2 -lrd 0.90
python bitwise_ss.py -exp cw_nocg -a identity -wb ste -ib ste -ub -lf net_tanh.model -cw -e 150 -bnm 0.2 -lrd 0.90
