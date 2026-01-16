MASTER_PORT=29506 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0,6 \
swift rlhf \
    --rlhf_type grpo \
    --model models/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --external_plugins ./scripts/grpo/plugin_adfm.py \
    --reward_funcs anomaly_cls_reward anomaly_loc_reward anomaly_cot_format \
    --use_liger_kernel \
    --train_type lora \
    --use_vllm true \
    --vllm_device cuda:1 \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --max_pixels 262144 \
    --vllm_limit_mm_per_prompt '{"image": 2}' \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset \
         data.jsonl \
    --max_completion_length 512 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 30 \
    --per_device_eval_batch_size 30 \
    --learning_rate 1e-05 \
    --warmup_ratio 0.001 \
    --gradient_accumulation_steps 10 \
    --eval_steps 10000 \
    --save_steps 20 \
    --save_total_limit 50 \
    --logging_steps 5 \
    --max_length 8192 \
    --padding_side left \
    --output_dir outputs/test \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 0.9 \
    --deepspeed ./scripts/deepspeed/ds_z2_config.json \
    --report_to tensorboard \
    --log_completions true 