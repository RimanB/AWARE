#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-3:00
#SBATCH --mem=3G
#SBATCH -o EFO_0003767_dandy-sweep-77_seed=10.out
#SBATCH -e EFO_0003767_dandy-sweep-77_seed=10.err

# conda activate aware

##########################################
# All celltypes vs. Global (Allocate 6 hours for full run; 3 hours for sweep)
##########################################

# Rheumatoid Arthritis (EFO_0000685)
#wandb agent michellemli/tx-target/miirgtyc --count 1
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 1 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 2 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 3 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 4 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 5 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 6 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 7 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 8 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 9 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 10 --num_epoch=2000

# Inflammatory bowel disease (EFO_0003767)
#wandb agent michellemli/tx-target/nqt0ygkc --count 1
#ruby-sweep-81
#Fraction of celltypes that outperform global: 0.6423841059602649 (97 out of 151)
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 1 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 2 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 3 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 4 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 5 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 6 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 7 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 8 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 9 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.5 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=1e-05 --random_state 10 --num_epoch=2000

#wandb agent michellemli/tx-target/cl1pyeen --count 1
#dandy-sweep-77
#Fraction of celltypes that outperform global: 0.6447368421052632 (98 out of 152)
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 1 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 2 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 3 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 4 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 5 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 6 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 7 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 8 --num_epoch=2000
#python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 9 --num_epoch=2000
python train.py --actn=relu --disease=EFO_0003767 --dropout=0.4 --embeddings_dir=/home/ml499/zitnik/scDrug/pipeline/AWARE/curious_sweep_1/ --globe global --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.001 --norm=ln --order=nd --wd=0.0001 --random_state 10 --num_epoch=2000
