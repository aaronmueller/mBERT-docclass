lang=$1
freeze=$2
gpu=0
if [[ $(hostname -f) = *clsp* ]]; then
    export PATH=$HOME/local/app/miniconda3/bin:$PATH
    gpu=`free-gpu`
fi

for bs in 16 32; do
for lr in 5e-5 3e-5 2e-5; do
for ep in 3.0 4.0; do
CUDA_VISIBLE_DEVICES=$gpu python src/run_classifier.py \
    --task_name tobacco \
    --lang $lang \
    --trg_lang ar,bn,de,en,es,fr,hi,id,pt,ru,ta,th,tr,uk,vi,zh \
    --do_train \
    --do_eval \
    --data_dir data/tobacco \
    --bert_model bert-base-multilingual-cased \
    --max_seq_length 128 \
    --train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $ep \
    --output_dir model/tobacco/freeze-lay$freeze/$lang/bs$bs-lr$lr-ep$ep \
    --freeze $freeze
done
done
done