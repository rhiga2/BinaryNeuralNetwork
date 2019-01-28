python binary_ae.py -ae -exp wae -ib identity -wb identity -e 256 -d 3 -lr 1e-4 -b 8 -k 2 -wavenet -l cel
python binary_ae.py -ae -exp bwae -ib clipped_ste -wb clipped_ste -e 256 -lr 1e-5 -d 3 -cw -b 8 -k 2 -l sdr -as -ug -wavenet -l cel
