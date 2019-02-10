python bitwise_dae.py -ae -exp wae -a tanh -ib identity -wb identity -e 256 -d 3 -lr 1e-4 -b 8 -model wavenet -l cel -ub
python bitwise_dae.py -ae -exp bwae -a tanh -ib clipped_ste -wb clipped_ste -e 256 -lr 1e-4 -d 3 -b 8 -as -model wavenet -l cel -lf wae.model -ub 
