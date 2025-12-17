# CausalMAE

This is the codebase for **CausalMAE: Causal Masked Autoencoder for Pathological Gait Classification** (submitted to [FCS - Frontiers of Computer Science, Manuscript ID: 252070](https://mc.manuscriptcentral.com/hepfcs)).

# Download Rec-G and NIH Datasets

The experimental datasets for pathological gait classification comprise two primary sources:

## 1. Rec-G
A merged dataset combining:
- **[GaitRec Dataset](https://springernature.figshare.com/collections/_/4788012)**
- **[Gutenberg Dataset](https://data.ncl.ac.uk/articles/dataset/Gutenberg_Dataset/24574753)** from Newcastle University

## 2. Gait in Parkinson's Disease (NIH)
A dataset curated by the National Institutes of Health (NIH):
- **[Gait in Parkinson's Disease Dataset](https://www.kaggle.com/datasets/zarif98sjs/gait-in-parkinsons-disease)**

Further details regarding these datasets are available via the provided links. For convenience, preprocessed versions of Rec-G and NIH are also accessible for direct training and evaluation:  
[UC (password: u9wn)](https://drive.uc.cn/s/abe8147adb2a4) | [Azure (password: 123)](http://www.jd.com) | [Google Drive (password: gd456)](http://www.jd.com)

# Download Pre-trained Models

[UC (password: SjR1)](https://drive.uc.cn/s/1b7b5a2fa5384) | [Azure (password: 567)](http://www.jd.com) | [Google Drive (password: gd123)](http://www.jd.com)

 * Phase I:

```
OUTPUT_DIR_PHASE1="./CausalMAE_master/output/pretrain_causalMae/phase1" OUTPUT_DIR_PHASE2="./CausalMAE_master/output/pretrain_causalMae/phase2"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=8675 --use_env run_causalMae_pretraining.py --phase 1 --mask_ratio 0.4  --batch_size 256 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 150 --output_dir_phase1 ${OUTPUT_DIR_PHASE1} --output_dir_phase2 ${OUTPUT_DIR_PHASE2} --log_dir ${LOG_DIR}
```
 * Phase II:

```
OUTPUT_DIR_PHASE1="./CausalMAE_master/output/pretrain_causalMae/phase1" OUTPUT_DIR_PHASE2="./CausalMAE_master/output/pretrain_causalMae/phase2"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=8675 --use_env run_causalMae_pretraining.py --phase 1 --mask_ratio 0.8  --batch_size 128 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 150 --output_dir_phase1 ${OUTPUT_DIR_PHASE1} --output_dir_phase2 ${OUTPUT_DIR_PHASE2} --log_dir ${LOG_DIR}

# Results

## Rec-G Results
This table summarizes our Rec-G results for CausalMAE:

| Dataset | Baselines | Acc | F1-Score | FLOPs |
|---------|-----------|-----|----------|-------|
| Rec-G | Baseline1 | xx | xx | xx |
| Rec-G | Baseline2 | xx | xx | xx |
| Rec-G | Baseline3 | xx | xx | xx |
| Rec-G | CausalMAE | xx | xx | xx |

## NIH Results
This table summarizes our NIH results for CausalMAE:

| Dataset | Baselines | Acc | F1-Score | FLOPs |
|---------|-----------|-----|----------|-------|
| NIH | Baseline1 | xx | xx | xx |
| NIH | Baseline2 | xx | xx | xx |
| NIH | Baseline3 | xx | xx | xx |
| NIH | CausalMAE | xx | xx | xx |
