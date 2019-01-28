# python binary_ae.py -ae -exp wae -a tanh -ib identity -wb identity -e 256 -d 3 -lr 1e-4 -b 8 -k 2 -wavenet -l cel
python binary_ae.py -ae -exp bwae -a tanh -ib clipped_ste -wb clipped_ste -e 256 -lr 1e-3 -d 3 -b 8 -k 2 -as -ug -wavenet -l cel -lf wae.model
