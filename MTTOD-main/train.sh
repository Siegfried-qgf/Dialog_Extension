CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch\
    --master_port 8888 \
    --nproc_per_node=8\
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    -data_type CRS\
    -num_gpus 8\
    -run_type train\
    -batch_size_per_gpu 2\
    -batch_size_per_gpu_eval 32\
    -model_dir ckpt/CRS_only_epoch20\
    -epochs 20\
    -seed 42\
    -add_auxiliary_task\
    -learning_rate 5e-4