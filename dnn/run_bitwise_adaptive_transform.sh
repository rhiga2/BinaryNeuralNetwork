python bitwise_dae.py -exp at -ib tanh -wb identity -e 250 -d 3 -lr 1e-3 -b 16 -l sisnr -ub -dropout 0.3
python bitwise_dae.py -exp bat -ib clipped_ste -wb clipped_ste -e 256 -d 3 -lr 1e-3 -b 16 -l sisnr -lf at.model -cw
