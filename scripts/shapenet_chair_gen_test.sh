#! /bin/bash
cate="chair"
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
data_dir="/mount/data"

python test.py \
    --cates ${cate} \
    --resume_checkpoint pretrained_models/3d_gen/${cate}/checkpoint-latest.pt \
    --use_latent_flow \
    --data_dir ${data_dir} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --zdim ${zdim} \


