# Cross-modal Information Flow in Multimodal Large Language Model
This is the official repository for the paper: [Cross-modal Information Flow in Multimodal Large Language Models](https://arxiv.org/abs/2411.18620)

# Installation
```
1. git clone https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM.git
2. cd cross-modal-information-flow-in-MLLM
3. Please following LLaVANEXT(https://github.com/LLaVA-VL/LLaVA-NeXT) to install the environment: llava
```
After installing llava environment, you will find an LLaVA-NeXT folder in cross-modal-information-flow-in-MLLM

# Dataset
Our dataset is collected from [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html). The collected datasets are in `datasets`. 
For the images, please download from [here](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip).

# Use
## Information flow
1. Open `scripts/informationFlow.sh`
2. Setting:
   
   `current_window`: how many layers you want to block for the attention knock at a time;
   
   `current_block_desc`: which kind of information flow you want to block;
   
   `model_path`: which kind of model you want to explore;
   
   `convmode`: different kind of model has different convmode;
   
   `dataset`: which kind of task you want to explore;
   
   `imagefolder`: the image folder
   
4. Run `sbatch scripts/informationFlow.sh`

**current_block_desc** can be chosen from:
```
  "Question->Last"
  "Image->Last"
  "Image->Question"
  "Last->Last"
  "Image Central Object->Question"
  "Image Without Central Object->Question"
```

**model_path** and **convmode** can be chosen from:
```
  model_path="liuhaotian/llava-v1.6-vicuna-7b" convmode="vicuna_v1"
  model_path="lmms-lab/llama3-llava-next-8b"  convmode="llava_llama_3"
  model_path="liuhaotian/llava-v1.5-7b"   convmode="vicuna_v1"
  model_path="liuhaotian/llava-v1.5-13b"   convmode="vicuna_v1"
```

**dataset** can be chosen from:
```
  datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv
  datasets/GQA_val_correct_question_with_positionQuery_QueryAttr.csv
  datasets/GQA_val_correct_question_with_existThatOr_LogicalObj.csv
  datasets/GQA_val_correct_question_with_twoCommon_CompareAttr.csv
  datasets/GQA_val_correct_question_with_relChooser_ChooseRel.csv
  datasets/GQA_val_correct_question_with_categoryThatThisChoose_objThisChoose_ChooseCat.csv
```

## Probability of answer word tracking
1. Open `scripts/last_position_answer_prob.sh`
2. Setting:
   
   `model_path`: which kind of model you want to explore;
   
   `convmode`: different kind of model has different convmode;
   
   `dataset`: which kind of task you want to explore;
   
   `imagefolder`: the image folder
   
4. Run `sbatch scripts/last_position_answer_prob.sh`

# Visulization
if you want to merge several lines into one figure, you can run `python vil/merge_lineplot.py`.

For example, you might already get the results of the information flow: `Question->Last`,`Image->Last`,`Last->Last`, and you want to merge these three lines in one Figure, and then you could run `python vil/merge_lineplot.py`.

## Cite
If this project is helpful for you, please cite our paper:
```
@article{zhang2024cross,
  title={Cross-modal Information Flow in Multimodal Large Language Models},
  author={Zhang, Zhi and Yadav, Srishti and Han, Fengze and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2411.18620},
  year={2024}
}
```


## Acknowledgement
The code is built upon https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions and [LLaVA](https://github.com/LLaVA-VL/LLaVA-NeXT).

Our used datasets are collected from [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html)
