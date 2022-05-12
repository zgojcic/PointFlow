#! /bin/bash

cate="car_new"
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=256
lr=2e-3
epochs=4000
ds=shapenet15k
log_name="gen/${ds}-cate_${cate}-seqback_dist"
data_dir="/raid/data"

python train.py \
    --log_name ${log_name} \
    --lr ${lr} \
    --train_T False \
    --dataset_type ${ds} \
    --data_dir ${data_dir} \
    --cates ${cate} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 40 \
    --viz_freq 10000 \
    --log_freq 1 \
    --val_freq 1000 \
    --distributed \
    --use_latent_flow

echo "Done"
exit 0
