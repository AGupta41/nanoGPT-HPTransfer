for depth in 8 
do
    for lr in 2e-3 #2e-5 2e-6 2e-7 2e-8 	
    do

    for seed in 1 #2 3
    do
    width=256
    head_size=64
    depth_alpha_exp=0.5
    n_heads=$((width / head_size))
    mup_base_depth=2
    mup_depth_multiplier=$(echo "scale=8; $depth/$mup_base_depth" | bc -l)
    out_dir="mup_examples/coord_check_shakespeare_char/sp/out/lamb_coord_check/depth${depth}_width256_lr${lr}_seed${seed}"
    python3.11 train_lamb.py \
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
        --max_iters=20 \
        --weight_decay=1e-1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --mup_enabled=False \
        #--mup_width_multiplier=1.0 \
        #--mup_input_alpha=1.0 \
        #--mup_output_alpha=1.0 \
        --mup_enable_coord_check_logging=True \
        #--depth_alpha_enabled=True  \
        #--depth_multiplier=$mup_depth_multiplier \
        #--depth_alpha_exp=$depth_alpha_exp \
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
