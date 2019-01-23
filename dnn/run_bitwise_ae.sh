# python bitwise_autoencoder.py -ae -exp ae -ib tanh -wb tanh -e 64
python bitwise_autoencoder.py -ae -exp bae -ib clipped_ste -wb clipped_ste -lf ae.model -e 256 -lr 1e-5
