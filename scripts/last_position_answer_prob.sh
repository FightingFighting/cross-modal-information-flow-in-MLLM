#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=llava-v1.5-13b
#SBATCH --output=output/slurm_output/slurm_%A_%a.out
#SBATCH --array=0


#models
#model_path="liuhaotian/llava-v1.6-vicuna-7b" convmode="vicuna_v1"
#model_path="lmms-lab/llama3-llava-next-8b"  convmode="llava_llama_3"
#model_path="liuhaotian/llava-v1.5-7b"   convmode="vicuna_v1"
model_path="liuhaotian/llava-v1.5-13b"   convmode="vicuna_v1"


#dataset
dataset=datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv
#dataset=datasets/GQA_val_correct_question_with_positionQuery_QueryAttr.csv
#dataset=datasets/GQA_val_correct_question_with_existThatOr_LogicalObj.csv
#dataset=datasets/GQA_val_correct_question_with_twoCommon_CompareAttr.csv
#dataset=datasets/GQA_val_correct_question_with_relChooser_ChooseRel.csv
#dataset=datasets/GQA_val_correct_question_with_categoryThatThisChoose_objThisChoose_ChooseCat.csv


imagefolder=../datasets/gqa/images/




#output name
model_name=$(basename "$model_path")
dataset_name=$(basename "$dataset" | sed 's/.*_//g' | sed 's/.csv//g')
job_id=$SLURM_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
output_file="output/slurm_output/${job_id}_${task_id}_model_${model_name}_dataset_${dataset_name}_only_cache.out"
#exec > "$output_file" 2>&1

echo "Output file: $output_file"
echo "Using model: $model_name"
echo "Using dataset: $dataset_name"

source activate llava
python last_position_answer_prob.py \
        --model-path $model_path \
        --image-folder $imagefolder \
        --temperature 0 \
        --conv-mode $convmode \
        --refined_dataset $dataset \
        --num_workers 2 \

