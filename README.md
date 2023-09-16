
## run

```sh
# nohup bash ./ab_bs.sh  >00.log 2>&1 &
GPUID=0
CONFIG=paras/ours.yaml
MODELCSEKG
NCLS=4
HDIM=64
BS=8192
DATA=ml-1m
for bs in 32768 8192 4096 2048 512 256; do
    python -u run.py --gpu_id=$GPUID --model=$MODEL --dataset=$DATA --config_files=$CONFIG \
    --context_hops=2 --n_factors=$NCLS --n_clusters=$NCLS --embedding_size=$HDIM \
    --node_dropout_rate=0.5 --learning_rate=0.0001 \
    --train_batch_size=$bs --eval_batch_size=40960 --eval_step=5
done

```

## plugin module
