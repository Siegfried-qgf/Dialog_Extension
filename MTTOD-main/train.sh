#3090
CUDA_VISIBLE_DEVICES=3 python main.py\
    -version 2.0\
    -data_type MUL\
    -num_gpus 1\
    -run_type train\
    -batch_size 4\
    -model_dir MUL_bsz4_ng1_aat_5e-4_5epoch_426_2\
    -epochs 5\
    -seed 42\
    -add_auxiliary_task\
    -learning_rate 5e-4