# Set the path to save checkpoints -
OUTPUT_DIR_PHASE1='./CausalMAE_master/output/pretrain_causalMae/phase1'
OUTPUT_DIR_PHASE2='./CausalMAE_master/output/pretrain_causalMae/phase2'
# Set the path to save TesnorBoard --log_dir
LOG_DIR='./CausalMAE_master/output/PT_results'

# LOG_FILE="PT_log_phase1_${TIMESTAMP}.txt"
# > PT_log_phase1.txt
# --batch_size 16 \
# --batch_size 128/64 \

# default 0.4 0.8

CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 --master_port=8675 --use_env run_causalMae_pretraining.py \
        --model pretrain_causalMae \
        --phase 2 \
        --mask_ratio 0.8 \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 100 \
        --output_dir_phase1 ${OUTPUT_DIR_PHASE1} \
        --output_dir_phase2 ${OUTPUT_DIR_PHASE2} \
        --log_dir ${LOG_DIR} \
        > PT_d_phase2.txt