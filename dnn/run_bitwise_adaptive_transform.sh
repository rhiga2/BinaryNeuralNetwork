python bitwise_dae.py -exp at -ib tanh -wb identity -e 200 -d 3 -lr 1e-2 -b 16 -l sisnr -dropout 0.2
python bitwise_dae.py -exp bat -ib clipped_ste -wb clipped_ste -e 256 -d 3 -lr 1e-3 -b 16 -l sisnr -lf at.model -cw
