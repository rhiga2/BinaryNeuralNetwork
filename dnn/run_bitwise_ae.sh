python bitwise_autoencoder.py -ae -exp ae -ib identity -wb identity -e 256 -d 0 -lr 1e-4 -b 16 -k 1024 -ug -wavenet -l bce
python bitwise_autoencoder.py -ae -exp bae -ib clipped_ste -wb clipped_ste -e 256 -lr 1e-5 -d 0 -cw -b 16 -k 1024 -l sdr -as -ug -wavenet -l bce
