# python bitwise_mnist.py -exp mnist -ib tanh -wb identity -e 200 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.1 -dp 150
# python bitwise_mnist.py -exp bmnist -ib clipped_ste -wb clipped_ste -lf mnist.model -cw -e 200 -b 128 -bnm 0.2 -lrd 0.1 -dp 100 -lr 1e-3
# python bitwise_mnist.py -exp mnist_tanh -ib tanh -wb tanh -e 200 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.1 -dp 150
# python bitwise_mnist.py -exp bmnist_tanh -ib clipped_ste -wb clipped_ste -lf mnist_tanh.model -cw -e 200 -b 128 -bnm 0.2 -lrd 0.1 -dp 100 -lr 1e-3
# python bitwise_mnist.py -exp mnist_signswiss -ib tanh -wb sign_swiss -e 200 -b 128 -lr 1e-3 -do 0.2 -bnm 0.01 -lrd 0.1 -dp 150
# python bitwise_mnist.py -exp bmnist_signswiss -ib clipped_ste -wb sign_swiss_ste -lf mnist_signswiss.model -cw -e 200 -b 128 -bnm 0.2 -lrd 0.1 -dp 100 -lr 1e-3
# python bitwise_mnist.py -exp mnist_ptanh -ib tanh -wb ptanh -e 200  -b 128 -lr 1e-3 -do 0.2 -bnm 0.01 -lrd 0.1 -dp 150
# python bitwise_mnist.py -exp bmnist_ptanh -ib clipped_ste -wb ptanh_ste -lf mnist_ptanh.model -cw -e 200 -b 128 -bnm 0.2 -lrd 0.1 -dp 100 -lr 1e-3
# python bitwise_mnist.py -exp mnist_htanh -ib tanh -wb htanh -e 200  -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.1 -dp 150
# python bitwise_mnist.py -exp bmnist_htanh -ib clipped_ste -wb clipped_ste -lf mnist_htanh.model -cw -e 200 -b 128 -bnm 0.2 -lrd 0.1 -dp 100 -lr 1e-3
python bitwise_mnist.py -exp mnist_phtanh -ib tanh -wb phtanh -e 200  -b 128 -lr 1e-3 -do 0.2 -bnm 0.01 -lrd 0.1 -dp 150
python bitwise_mnist.py -exp bmnist_phtanh -ib clipped_ste -wb phtanh_ste -lf mnist_phtanh.model -cw -e 200 -b 128 -bnm 0.2 -lrd 0.1 -dp 100 -lr 1e-3
