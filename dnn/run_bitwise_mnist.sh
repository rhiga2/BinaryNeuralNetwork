# python bitwise_mnist.py -exp mnist -ib tanh -wb identity -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp mnist_tanh -ib tanh -wb tanh -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp bmnist_noinit -ib tanh_ste -wb tanh_ste -e 300 -b 128 -cw -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp bmnist -ib tanh_ste -wb tanh_ste -e 300 -lf mnist.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp bmnist_tanh -ib tanh_ste -e 300 -wb tanh_ste -lf mnist_tanh.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100

# python bitwise_mnist.py -exp mnist_signswiss -ib tanh -wb sign_swiss -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp mnist_ptanh -ib tanh -wb ptanh -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp mnist_htanh -ib tanh -wb htanh -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100
# python bitwise_mnist.py -exp mnist_phtanh -ib tanh -wb phtanh -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100

python bitwise_mnist.py -exp bmnist_signswiss -ib tanh_ste -wb sign_swiss_ste -lf mnist_signswiss.model -cw -e 300 -b 128 -bnm 0.2 -lr 5e-4 -lrd 0.3 -dp 100
python bitwise_mnist.py -exp bmnist_ptanh -ib tanh_ste -wb ptanh_ste -lf mnist_ptanh.model -cw -e 300 -b 128 -bnm 0.2 -lr 5e-4 -lrd 0.3 -dp 100
python bitwise_mnist.py -exp bmnist_htanh -ib tanh_ste -wb clipped_ste -lf mnist_htanh.model -cw -e 300 -b 128 -bnm 0.2 -lr 5e-4 -lrd 0.3 -dp 100
python bitwise_mnist.py -exp bmnist_phtanh -ib tanh_ste -wb phtanh_ste -lf mnist_phtanh.model -cw -e 300 -b 128 -bnm 0.2 -lr 5e-4 -lrd 0.3 -dp 100

python bitwise_mnist.py -exp bmnist_25 -ib tanh_ste -wb tanh_ste -e 300 -lf mnist.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -sparsity 25
python bitwise_mnist.py -exp bmnist_33 -ib tanh_ste -wb tanh_ste -e 300 -lf mnist.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -sparsity 33
python bitwise_mnist.py -exp bmnist_50 -ib tanh_ste -wd tanh_ste -e 300 -lf mnist.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -sparsity 50
python bitwise_mnist.py -exp mnist_ug -ib tanh -wb identity -e 300 -b 128 -lr 1e-2 -do 0.2 -bnm 0.01 -lrd 0.3 -dp 100 -ug
python bitwise_mnist.py -exp bmnist_ug -ib tanh_ste -wb tanh_ste -e 300 -lf mnist_ug.model -cw -b 128 -bnm 0.2 -lr 1e-3 -lrd 0.3 -dp 100 -ug
