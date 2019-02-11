python bitwise_dae.py -exp at -ib identity -wb identity -e 150 -d 0 -lr 1e-3 -d 3 -b 16 -l sisnr -ub -dropout 0.3
python bitwise_dae.py -exp bat -ib clipped_ste -wb clipped_ste -e 256 -d 0 -lr 1e-3 -d 3 -b 16 -l sisnr -ub -lf ae.model -dropout 0 -lrd 0.9 -as -cw
