# experiement name -exp string
# epoch -e int
# batchsize -b int
# device -d int
# toy -toy
# load_file -lf string
# learning_rate -lr float
# weight_decay -wd
# dropout -d float
# period -p int
# loss -l string
# weighted -w
# sparsity -s float
# use_gate -ug
# use_batchnorm -ub
# bn_momentum -bnm
# activation -a string
# clip_weights -cw

python bitwise_ss.py -exp net_prelu -e 128 -a prelu
