python bitwise_autoencoder.py -ae -exp ae -ib tanh -wb tanh -e 256
python bitwise_ss.py -ae -exp bae -ib clipped_ste -wb clipped_ste -lf ae.model -e 128
