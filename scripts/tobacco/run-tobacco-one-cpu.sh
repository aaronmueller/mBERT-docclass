#source /home/shijie/local/app/miniconda3/bin/activate
source /home/amueller/miniconda3/bin/activate
conda activate pytorch_cpu

lang=$1
#gpu=0
#if [[ $(hostname -f) = *clsp* ]]; then
#    export PATH=~shijie/local/app/miniconda3/bin:$PATH
#    gpu=`free-gpu`
#fi

#export CUDA_VISIBLE_DEVICES=$gpu
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/shijie/local/cuda/lib64

#for bs in 16 32; do
#for lr in 5e-5 3e-5 2e-5; do
#for ep in 3.0 4.0 6.0 8.0 12.0 16.0; do
for bs in 32; do
for lr in 5e-5; do
for ep in 12.0; do
time python src/run_classifier.py \
    --task_name tobacco-feedback \
    --lang all \
    --trg_lang all \
    --do_train \
    --do_eval \
    --data_dir data/tobacco/clean_feedback_new.json \
    --bert-mode bert-base-multilingual-cased \
	--encoder bert \
	--decoder pool \
    --max_seq_length 128 \
    --train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $ep \
	--no_cuda \
    --output_dir model/tobacco-new/tuneall/one-model/bs$bs-lr$lr-ep$ep
done
done
done
# --separate_top
# $bs, $lr, $ep
