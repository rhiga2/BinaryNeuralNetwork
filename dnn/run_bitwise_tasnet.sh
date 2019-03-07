python bitwise_dae.py -exp tasnet -ib tanh -wb identity -e 64 -d 0 -lr 1e-3 -lrd 0.1 -dp 32 -b 4 -model tasnet -l sisnr
python bitwise_dae.py -exp btasnet -ib clipped_ste -wb clipped_ste -e 64 -d 0 -lr 1e-2 -lrd 0.1 -dp 32 -b 4 -model tasnet -lf tasnet.model -l sisnr
