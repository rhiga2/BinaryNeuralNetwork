python bitwise_ss.py -exp relu_ss -ba identity -e 150 -b 32  -dropout 0.3 -bnm 0.01 -lr 1e-4
python bitwise_ss.py -exp cw_cg_ss -ba clipped_ste -lf relu_ss.model -cw -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-5
python bitwise_ss.py -exp cw_nocg_ss -ba ste -lf relu_ss.model -cw -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-5
python bitwise_ss.py -exp nocw_nocg_ss -ba ste -lf relu_ss.model -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-5
python bitwise_ss.py -exp nocw_cg_ss -ba clipped_ste -lf relu_ss.model -e 150 -b 32 -bnm 0.2 -lrd 0.90 -lr 1e-5
