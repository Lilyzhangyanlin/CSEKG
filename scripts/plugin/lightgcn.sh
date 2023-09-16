# @0518
# nohup bash ./runs/plugin/lightgcn.sh >03.log 2>&1 &
GPUID=3
CONFIG=paras/lightgcn.yaml
# MODEL=CS_LightGCN
NCLS=4
NLAYERS=2
HDIM=64
# ml-1m, amazon-book2, lfm-small
DATA=amazon-book2
# DATA=ml-1m 
# DATA=lfm-small
# for model in CS_LightGCN LightGCN; do
for model in CS_LightGCN_Cls CS_LightGCN_Imp LightGCN; do
    python -u run.py --gpu_id=$GPUID --model=$model --dataset=$DATA --config_files=$CONFIG \
    --embedding_size=$HDIM --n_layers=$NLAYERS \
    --n_clusters=$NCLS \
    --learning_rate=0.0001 --eval_step=5 
    # --train_batch_size=8192 --eval_batch_size=40960 
done



