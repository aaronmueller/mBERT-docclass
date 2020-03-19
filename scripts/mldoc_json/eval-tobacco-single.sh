source /home/shijie/local/app/miniconda3/bin/activate

lang=$1
gpu=0
if [[ $(hostname -f) = *clsp* ]]; then
    export PATH=~shijie/local/app/miniconda3/bin:$PATH
    gpu=`free-gpu`
fi

export CUDA_VISIBLE_DEVICES=$gpu
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/shijie/local/cuda/lib64

for lang in $(ls mldoc-langs/); do
CUDA_VISIBLE_DEVICES=$gpu
python src/run_classifier.py \
    --task_name mldoc-json \
    --lang $lang \
    --trg_lang $lang \
    --do_eval \
    --data_dir data/mldoc_json \
    --bert-mode bert-base-multilingual-cased \
	--encoder bert \
	--decoder pool \
    --max_seq_length 128 \
    --load model/mldoc/tuneall/$lang/best/model.pth \
    --output_dir eval/mldoc/tuneall/$lang-$lang \
    --no_eval_dev
done

# change --load back to $lang from en!!!
