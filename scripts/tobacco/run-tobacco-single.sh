source /home/shijie/local/app/miniconda3/bin/activate

lang=$1
gpu=0
if [[ $(hostname -f) = *clsp* ]]; then
    export PATH=~shijie/local/app/miniconda3/bin:$PATH
    gpu=`free-gpu`
fi

export CUDA_VISIBLE_DEVICES=$gpu
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/shijie/local/cuda/lib64

for bs in 16 32; do
for lr in 5e-5 3e-5 2e-5; do
for ep in 3.0 4.0 6.0 8.0 12.0 16.0; do
export CUDA_VISIBLE_DEVICES=$gpu
python /export/a05/shijie/bert/src/run_classifier.py \
    --task_name tobacco \
    --lang $lang \
    --trg_lang $lang \
    --do_train \
    --do_eval \
    --data_dir /export/a05/shijie/bert/data/tobacco \
    --max_seq_length 128 \
    --train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $ep \
	--bert-mode bert-base-multilingual-cased \
	--encoder bert \
	--decoder pool \
    --output_dir model/tobacco/tuneall/$lang/bs$bs-lr$lr-ep$ep
done
done
done
