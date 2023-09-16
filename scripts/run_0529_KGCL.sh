# @0519
# nohup bash run.sh >01.log 2>&1 &

# MODEL=KGCL
# CONFIG=paras/kgcl.yaml
MODEL=KGRec
CONFIG=paras/kgrec.yaml
for DATA in lfm-small ml-1m amazon-book2; do
    python3 -u run.py --model=$MODEL --dataset=$DATA --config_files=$CONFIG
done
