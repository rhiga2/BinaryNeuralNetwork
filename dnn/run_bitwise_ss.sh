python bitwise_ss.py -exp relu -a relu -wb identity -ib identity -ub -e 150 -dropout 0.3
python bitwise_ss.py -exp cw_cg -a relu -wb clipped_ste -ib clipped_ste -ub -lf relu.model -cw -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4
python bitwise_ss.py -exp cw_nocg -a relu -wb ste -ib ste -ub -lf relu.model -cw -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4
python bitwise_ss.py -exp nocw_nocg -a relu -wb ste -ib ste -ub -lf relu.model -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4
python bitwise_ss.py -exp nocw_cg -a relu -wb clipped_ste -ib clipped_ste -ub -lf relu.model -e 150 -bnm 0.2 -lrd 0.90 -lr 1e-4
