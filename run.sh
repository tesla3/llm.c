torchrun \
	--standalone \
	--nproc_per_node=8 \
	train_gpt2.py \
	--input_bin=dev/data/fineweb10B/fineweb_train_*.bin \
  --input_val_bin=dev/data/fineweb10B/fineweb_val_*.bin \
	--write_tensors=0 \
	--model=d12 \
	--batch_size=32 \
	--sequence_length=1024 \
	--total_batch_size=524288 \
	--dtype=bfloat16 \
	--compile=1 \
	--tensorcores=1 \
	--flash=1 \
	--num_iterations=18865 \
  --zero_stage=1  \
	--weight_decay=0.1 \
  --learning_rate=0.0006 \
  --val_loss_every=256 \
  --val_max_steps=64 \
  --overfit_single_batch=0 \
  2>&1 \
  |tee train.log 


<<\c
torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_gpt2.py \
  --total_batch_size=1048576 \
  --write_tensors=0 \
  --num_iterations=64 \
  --sequence_length=1024 \
  --compile=1 \
  --tensorcores=1 \
  --dtype=bfloat16 \
  --model=d12 \
  --zero_stage=1 \
  --input_bin=dev/data/tinystories/TinyStories_train.bin \
  --input_val_bin=dev/data/tinystories/TinyStories_val.bin
c
