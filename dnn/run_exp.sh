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
# bn_momentum -bnm float
# activation -a string
# clip_weights -cw

python bitwise_ss.py -exp net_tanh -ub
python bitwise_ss.py -exp net_stetanh -wa ste_tanh -ub
python bitwise_ss.py -exp net_identity -wa identity -ub
python bitwise_ss.py -exp net_clippedste_noinit -a clippedste -wa clippedste -cw -ub
python bitwise_ss.py -exp net_clippedste_tanh -a clippedstr -wa clippedste -cw -lf net_tanh.model -ub
python bitwise_ss.py -exp net_clippedste_stetanh -a clippedste -wa clippedste -cw -lf net_stetanh.model -ub
python bitwise_ss.py -exp net_clippedste_identity -a clippedstr -wa clippedste -cw -lf net_identity.model -ub
