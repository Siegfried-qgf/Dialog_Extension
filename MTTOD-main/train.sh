CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_port 8888 \
    --nproc_per_node=8\
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    -data_type MUL\
    -version 2.0 \
    -num_gpus 8\
    -run_type train\
    -batch_size_per_gpu 4\
    -model_dir MUL_5.7\
    -epochs 5\
    -seed 42\
    -learning_rate 2e-3