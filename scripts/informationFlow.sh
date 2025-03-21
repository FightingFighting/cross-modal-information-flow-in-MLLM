#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --job-name=llava-v1.5-13b
#SBATCH --output=output/slurm_output/slurm_%A_%a.out
#SBATCH --array=0

current_window=9

#"Question->Last"
#"Image->Last"
#"Image->Question"
#"Last->Last"
#"Image Central Object->Question"
#"Image Without Central Object->Question"
current_block_desc="Question->Last"


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

imagefold=/gpfs/work5/0/prjs0370/zhizhang/projects/Dissection/datasets/gqa/images/



#output name
save_block_desc=$(echo "$current_block_desc" | sed 's/[ ]/_/g' | sed 's/->/__/g' | sed 's/[^a-zA-Z0-9_]/_/g')
model_name=$(basename "$model_path")
dataset_name=$(basename "$dataset" | sed 's/.*_//g' | sed 's/.csv//g')
job_id=$SLURM_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
output_file="output/slurm_output/${job_id}_${task_id}_window_${current_window}_block_${save_block_desc}_model_${model_name}_dataset_${dataset_name}.out"
#exec > "$output_file" 2>&1

echo "Output file: $output_file"
echo "Running job with window size: $current_window"
echo "Running job with block description: $current_block_desc"
echo "Using model: $model_name"
echo "Using dataset: $dataset_name"



source activate llava
python InformationFlow.py \
        --model-path $model_path \
        --image-folder $imagefold \
        --temperature 0 \
        --conv-mode $convmode \
        --refined_dataset $dataset \
        --window $current_window \
        --block_description "$current_block_desc" \
        --num_workers 2 \

