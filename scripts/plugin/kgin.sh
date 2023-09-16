# @0519
# nohup bash ./runs/plugin/kgin.sh >01.log 2>&1 &
GPUID=1
CONFIG=paras/kgin.yaml
NCLS=4

HDIM=64
NHOPS=2
IND=cosine
# ml-1m, amazon-book2, lfm-small
DATA=lfm-small
# DATA=ml-1m 
# DATA=amazon-book2
for model in CS_KGIN_Cls CS_KGIN_Imp KGIN; do
    python -u run.py --gpu_id=$GPUID --model=$model --dataset=$DATA --config_files=$CONFIG \
    --embedding_size=$HDIM  --context_hops=$NHOPS --ind=$IND \
    --n_clusters=$NCLS \
    --learning_rate=0.001 --eval_step=5 
    # --layers=$LAYERS
    # --train_batch_size=8192 --eval_batch_size=40960 
done



