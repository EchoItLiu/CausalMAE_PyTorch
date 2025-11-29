# Set the path to save checkpoints - 
OUTPUT_DIR='./CausalMAE_master/output/finetune_causalMae'
# path to pretrain model, load mask_phase2_checkpoint
MODEL_PATH='./CausalMAE_master/output/pretrain_causalMae/phase2/checkpoint-99.pth'
# path to tensorboard
LOG_DIR='./CausalMAE_master/output/FT_results'
# path to confusion matrix
CM_DIR='./CausalMAE_master/output/cfm'

# BS: 8 → 16 → 32 → 48

CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 --master_port=8675 --use_env run_causalMae_finetuning.py \
    --model finetune_causalMae \
    --phase 3 \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --cm_dir ${CM_DIR} \
    --batch_size 32 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    > FT_log_d.txt
    