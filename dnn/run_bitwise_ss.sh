python bitwise_ss.py -exp tanh -a relu -wb identity -ib identity -ub -e 150 -dropout 0.3
python bitwise_ss.py -exp nocw_nocg -a relu -wb ste -ib ste -ub -lf net_tanh.model -e 150 -bnm 0.2 -lrd 0.90 -as
python bitwise_ss.py -exp nocw_cg -a relu -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -e 150 -bnm 0.2 -lrd 0.90 -as
python bitwise_ss.py -exp cw_cg -a relu -wb clipped_ste -ib clipped_ste -ub -lf net_tanh.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as
python bitwise_ss.py -exp cw_nocg -a relu -wb ste -ib ste -ub -lf net_tanh.model -cw -e 150 -bnm 0.2 -lrd 0.90 -as
