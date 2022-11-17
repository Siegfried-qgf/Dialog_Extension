CUDA_VISIBLE_DEVICES=1 python main.py\
    -ckpt ./ckpt/MUL_fewshot_qa_ccmask_50epoch_2/ckpt-epoch50\
    -run_type predict\
    -batch_size_per_gpu_eval 4\
    -data_type QA\
    -output D2D\
    -train_subnet







