CUDA_VISIBLE_DEVICES=4 python main.py\
    -ckpt ./ckpt/CRS_only_epoch20/ckpt-epoch20\
    -run_type predict\
    -batch_size_per_gpu_eval 32\
    -data_type CRS\
    -output CRS





