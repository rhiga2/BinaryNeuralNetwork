python bitwise_dae.py -exp at -a tanh -ib tanh -wb tanh -e 32 -d 0 -lr 1e-3 -d 3 -b 8 -l sdr -ub -ug -dropout 0.2
python bitwise_dae.py -exp bat -a clipped_ste -ib clipped_ste -wb clipped_ste -e 256 -d 0 -lr 1e-4 -d 3 -b 8 -l sdr -ub -ug -lf ae.model -dropout 0 -lrd 0.9
