
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --master_port 8888 \
    --nproc_per_node=4\
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    -data_type TOD\
    -version 2.0 \
    -num_gpus 4\
    -run_type train\
    -batch_size_per_gpu 4\
    -model_dir testddp_tod\
    -epochs 5\
    -seed 42 \
    -learning_rate 2e-3