#3090
CUDA_VISIBLE_DEVICES=0 python main.py\
    -ckpt ./MUL_bsz4_ng1_aat_5e-4_5epoch_426_qadel/ckpt-epoch5\
    -run_type predict\
    -batch_size 4\
    -data_type CRS\
    -output CRS





