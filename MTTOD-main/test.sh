CUDA_VISIBLE_DEVICES=0 python main.py\
    -ckpt ./ckpt/QA_Squad_ConvQA_10epoch/ckpt-epoch10\
    -run_type predict\
    -batch_size_per_gpu_eval 64\
    -data_type CC_UB\
    -output CC_UB\
    -search_subnet





