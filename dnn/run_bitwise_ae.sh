python bitwise_autoencoder.py -ae -exp ae -ib identity -wb identity -e 64 -d 3 -lr 1e-4 -b 16 -k 1024 
python bitwise_autoencoder.py -ae -exp bae -ib clipped_ste -wb clipped_ste -lf ae.model -e 256 -lr 1e-5 -d 3 -cw -b 16 -k 1024 -l sdr
