source /home/shijie/local/app/miniconda3/bin/activate

lang=$1
gpu=0
if [[ $(hostname -f) = *clsp* ]]; then
    export PATH=~shijie/local/app/miniconda3/bin:$PATH
    gpu=`free-gpu`
fi

export CUDA_VISIBLE_DEVICES=$gpu
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/shijie/local/cuda/lib64

#for bs in 16 32; do
for lr in 5e-5 3e-5 2e-5; do
for ep in 3.0 4.0 6.0 8.0 12.0 16.0; do
#for lr in 2e-5; do
#for ep in 6.0 8.0 12.0 16.0; do
#for ep in 3.0 4.0; do
CUDA_VISIBLE_DEVICES=$gpu
python src/run_classifier.py \
    --task_name mldoc-json \
    --lang all \
    --trg_lang de,en,es,fr,it,ja,ru,zh \
	--separate_top \
    --do_train \
    --do_eval \
    --data_dir data/mldoc_json/combined_mldoc.json \
    --bert-mode bert-base-multilingual-cased \
	--encoder bert \
	--decoder pool \
    --max_seq_length 128 \
    --train_batch_size 1 \
	--eval_batch_size 1 \
    --learning_rate $lr \
    --num_train_epochs $ep \
    --output_dir model/mldoc/tuneall/one-model-sepclass/bs1-lr$lr-ep$ep
done
done
# done
# replace 1 with $bs after done!!
# --separate_top
