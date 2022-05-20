CUDA_VISIBLE_DEVICES=2 python main.py\
    -ckpt ./ckpt/Ubuntu_epoch10/ckpt-epoch10\
    -run_type predict\
    -batch_size_per_gpu_eval 64\
    -data_type CC_UB\
    -output CC2





