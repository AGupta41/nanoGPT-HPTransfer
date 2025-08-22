for depth in 2 4 8 16 32 64
do
    for lr in 4.44e-2 #2e-9 2e-10 	
    do

    for seed in 1 2 3
    do
    width=256
    head_size=64
    depth_alpha_exp=1.0
    n_heads=$((width / head_size))
    mup_base_depth=2
    mup_depth_multiplier=$(echo "scale=8; $depth/$mup_base_depth" | bc -l)
    out_dir="mup_examples/coord_check_shakespeare_char/mup/depth_alpha_1/out/adopt_early_stop/depth${depth}_width256_lr${lr}_seed${seed}"
    python3 train_adopt.py \
        --out_dir=$out_dir \
        --eval_interval=1 \
        --log_interval=1 \
        --eval_iters=1 \
        --eval_only=False \
        --always_save_checkpoint=False \
        --never_save_checkpoint=True \
        --init_from='scratch' \
        --wandb_log=False \
        --csv_log=True \
        --dataset='shakespeare_char' \
        --gradient_accumulation_steps=4 \
        --batch_size=2 \
        --block_size=1024 \
        --n_layer=$depth \
        --n_head=$n_heads \
        --n_embd=$width \
        --dropout=0.0 \
        --bias=True \
        --init_std=0.02 \
        --learning_rate=$lr \
        --max_iters=15000 \
        --weight_decay=1e-1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --mup_enabled=True \
        --mup_width_multiplier=1.0 \
        --mup_input_alpha=1.0 \
        --mup_output_alpha=1.0 \
        --mup_enable_coord_check_logging=True \
        --depth_alpha_enabled=True  \
        --depth_multiplier=$mup_depth_multiplier \
        --depth_alpha_exp=$depth_alpha_exp \
        --seed=$seed \
	--rho=0.03\
        --backend='nccl' \
        --device='cuda' \
        --dtype='float32' \
        --compile=False
    done
    done
done
        # --device='cuda' \
        # --device='cpu' \
