accelerate launch --mixed_precision="bf16" train_text_to_image_lora_sd3.py \
  --pretrained_model_name_or_path=/root/autodl-tmp/sd3_medium_incl_clips_t5xxlfp16.safetensors \
  --instance_data_dir /root/autodl-tmp/liewen_norm \
  --image_column="image" \
  --caption_column="text" \
  --instance_prompt="defect of crack" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=60 \
  --checkpointing_steps=100000000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-ft-model-lora-crack-bf16" \
  --mixed_precision="bf16" \
  --rank=64
  # --validation_prompt="defect of crack"