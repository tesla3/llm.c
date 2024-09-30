#!/bin/bash

# the same as scripts/run_gpt2_124M.sh but with PyTorch

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_gpt2.py ... (all the other arguments the same)
#torchrun --standalone --nproc_per_node=8 train_gpt2.py \
python train_gpt2.py \
    --input_bin "dev/data/fineweb100B/fineweb_train_*.bin" \
    --input_val_bin "dev/data/fineweb100B/fineweb_val_*.bin" \
    --output_dir pylog_gpt2_124M \
    --val_loss_every 250 \
    --sample_every 0 \
    --write_tensors 0 \
    --model d12 \
    --batch_size 16 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 100 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --warmup_iters 0 \
    --learning_rate_decay_frac 1.0 \
    --overfit_single_batch 1

