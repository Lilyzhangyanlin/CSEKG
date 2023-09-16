# @0518
# nohup bash ./runs/plugin/kgat.sh >02.log 2>&1 &
GPUID=2
CONFIG=paras/kgat.yaml
NCLS=4
# LAYERS=(64 32)
HDIM=64
AGG=bi
# ml-1m, amazon-book2, lfm-small
# DATA=ml-1m 
# DATA=lfm-small
DATA=amazon-book2
for model in CS_KGAT_Cls CS_KGAT_Imp KGAT; do
# for model in KGAT; do
    python -u run.py --gpu_id=$GPUID --model=$model --dataset=$DATA --config_files=$CONFIG \
    --embedding_size=$HDIM --kg_embedding_size=$HDIM  --aggregator_type=$AGG \
    --n_clusters=$NCLS \
    --learning_rate=0.001 --eval_step=5 
    # --layers=$LAYERS
    # --train_batch_size=8192 --eval_batch_size=40960 
done



