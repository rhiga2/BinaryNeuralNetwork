python bitwise_ss.py -exp relu_ss -a relu -ba identity -e 150 -dropout 0.3
python bitwise_ss.py -exp cw_cg_ss -a relu -ba clipped_ste -lf relu_ss.model -cw -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4 -as
python bitwise_ss.py -exp cw_nocg_ss -a relu -ba ste -lf relu_ss.model -cw -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4 -as
python bitwise_ss.py -exp nocw_nocg_ss -a relu -ba ste -lf relu_ss.model -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4 -as
python bitwise_ss.py -exp nocw_cg_ss -a relu -ba clipped_ste -lf relu_ss.model -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4 -as
