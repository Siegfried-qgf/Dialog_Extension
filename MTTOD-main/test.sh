#3090
CUDA_VISIBLE_DEVICES=1 python main.py\
    -ckpt ./MUL_5.6/ckpt-epoch5\
    -run_type predict\
    -batch_size 4\
    -data_type CRS\
    -output CRS_goal\





