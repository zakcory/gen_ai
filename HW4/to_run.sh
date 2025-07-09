#!/bin/bash

conda activate cs236207

# python image_diffusion_todo/train.py --log_interval 5000 --batch_size 8 --train_num_steps 150000 --use_cfg

# echo "FINISHED TRAINING ----------------------- STARTING SAMPLING"

# echo "------------RUNNING WITH 7.5 SCALE----------------"
# python image_diffusion_todo/sampling.py --ckpt_path results/cfg_diffusion-ddpm-07-09-112121/last.ckpt --save_dir sampled_imgs_scale_7p5 --use_cfg 

# echo "------------RUNNING WITH 0.0 SCALE----------------"
# python image_diffusion_todo/sampling.py --ckpt_path results/cfg_diffusion-ddpm-07-09-112121/last.ckpt --save_dir sampled_imgs_scale_0 --cfg_scale 0.0 --use_cfg  

echo "--------------RUNNING DATASET PREPROCESSING-----------"
python image_diffusion_todo/dataset.py
