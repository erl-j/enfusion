export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="artefacts/scaled_aesd_dataset/"

accelerate launch --mixed_precision="fp16" train_text_to_image_modified.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-scaled-asdf"