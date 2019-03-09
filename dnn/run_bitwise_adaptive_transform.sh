python bitwise_dae.py -exp at -ib tanh -wb identity -e 64 -d 3 -lr 1e-2 -lrd 0.1 -dp 64 -b 32 -l sisnr -dropout 0
python bitwise_dae.py -exp bat -ib clipped_ste -wb clipped_ste -e 64 -d 3 -lr 1e-3 -lrd 0.1 -dp 64 -b 32 -l sisnr -lf at.model -cw
