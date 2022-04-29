#3090
CUDA_VISIBLE_DEVICES=1 python main.py\
    -ckpt ./testddp_tod/ckpt-epoch5\
    -run_type predict\
    -batch_size 4\
    -data_type TOD\
    -output TOD





