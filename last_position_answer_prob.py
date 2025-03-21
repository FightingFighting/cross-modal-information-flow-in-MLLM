from InformationFlow import create_data_loader

import copy
import pdb

from methods import *

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns

import argparse
import os
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path




def generate_plot_attrscore(data, save_file, x="layer", ys="", layer_num=0):
    hex_colors = ["#f20089", "#5c95ff", "#ffa9a3", "#b9e6ff" ]
    palette = sns.color_palette(hex_colors)

    sns.set(context="notebook")
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(4, 4))

    ax = sns.lineplot(data, x=x, y=ys[0],
                      label=ys[0],color=palette[0],
                      dashes=False,
                      linewidth=3)

    for ind, y in enumerate(ys[1:]):
        sns.lineplot(data, x=x, y=y,
                     label=y,color=palette[ind+1],
                     dashes=False,
                     linewidth=3)


    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, layer_num + 0.5)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.legend(fontsize=6,handlelength=1)

    plt.savefig(save_file)




def run_original(model, inps, tokenizer, model_name):
    with torch.inference_mode():
        output_details = model.generate(**inps)

    answer_token_id = output_details['sequences']

    first_answer_hidden_id=0

    predicted_answer = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip().lower()



    hs_alllayer_first_answer_gen=[]
    for layer_id in range(model.config.num_hidden_layers+1):
        hs_first_answer_gen = output_details['hidden_states'][first_answer_hidden_id][layer_id][:,-1,:].squeeze().cpu() #torch.Size([4096])
        hs_alllayer_first_answer_gen.append(hs_first_answer_gen)
    return hs_alllayer_first_answer_gen, predicted_answer




def cache_hiddenstate(data_loader, questions, model, tokenizer, dataset_dict, model_name):
    # Run attention knockouts
    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    hs_cache_first_answer_gen_all = {}
    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor), line in tqdm(zip(data_loader, questions),total=len(questions)):

        question_id = line["q_id"]
        img_id=line["img_id"]


        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        inps = {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": original_image_sizes,
            "do_sample": True if args.temperature > 0 else False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
            "output_hidden_states":True

        }

        answer = dataset_dict[question_id]["answer"]


        hs_cache_first_answer_gen, predicted_answer = run_original(model, inps,tokenizer,model_name)

        if answer != predicted_answer:
            continue

        hs_cache_first_answer_gen_all[question_id]={}
        for layer in layers_to_cache:
            hs_cache_first_answer_gen_all[question_id][(question_id, img_id, layer)]=hs_cache_first_answer_gen[layer]

    return hs_cache_first_answer_gen_all


# Information flow analysis
def main(args):


    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
                                                                          args.model_base,
                                                                          model_name,
                                                                          device_map="auto",
                                                                          attn_implementation=None)
    model.eval()
    model.tie_weights()


    #dataset
    #predict correct and filter
    task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
    df = pd.read_csv(args.refined_dataset, dtype={"question_id":str}).fillna('')
    dataset_dict = df.set_index('question_id').T.to_dict('dict')
    questions = [ {**detail, "q_id":qu_id} for qu_id, detail in dataset_dict.items()]
    data_loader = create_data_loader(questions, args.image_folder,  args.batch_size, args.num_workers, tokenizer,  image_processor, model.config, task_name, args.conv_mode)



    if args.only_read_cache:
        file_name = f"cache_hiddenFeature"
        cache_path=f"output/temp/last_position_answer_probs/{model_name}/{task_name}/val/{file_name}.npy"
        print(f"read files form here: {cache_path}", flush=True)
        hs_cache_first_answer_gen_all = np.load(cache_path, allow_pickle=True).tolist()
    else:
        #cashe hidden state
        hs_cache_first_answer_gen_all = cache_hiddenstate(data_loader, questions, model, tokenizer, dataset_dict, model_name)
        if args.only_cache:
            file_name = "cache_hiddenFeature"
            os.makedirs(f"output/temp/last_position_answer_probs/{model_name}/{task_name}/val", exist_ok=True)
            np.save(f"output/temp/last_position_answer_probs/{model_name}/{task_name}/val/{file_name}.npy",hs_cache_first_answer_gen_all)
            exit(0)





    records = []
    E = model.get_output_embeddings().weight.to(torch.float32).cpu().detach()
    for line in tqdm(questions,total=len(questions)):


        question_id = line["q_id"]
        img_id=line["img_id"]

        if question_id not in hs_cache_first_answer_gen_all: continue

        question = dataset_dict[question_id]["question"]
        answer = dataset_dict[question_id]["answer"].lower()
        if task_name == "ChooseRel" or  task_name == "ChooseAttr" or task_name == "ChooseCat":
            true_option = dataset_dict[question_id]["true option"]
            false_option = dataset_dict[question_id]["false option"]


        hs_cache_first_answer_gen_question=hs_cache_first_answer_gen_all[question_id]
        for layer in range(model.config.num_hidden_layers+1):
            hs_first_generated_token = hs_cache_first_answer_gen_question[(question_id, img_id, layer)].cpu().to(torch.float32)
            logits_first_generated_token = hs_first_generated_token.matmul(E.T)
            scores_first_generated_token = torch.softmax(logits_first_generated_token, dim=-1).numpy()

            top_k = [(tokenizer.decode([i]), i, scores_first_generated_token[i]) for i in np.argsort(-scores_first_generated_token)[:50]]
            top_k_word, top_k_token, top_k_score = zip(*top_k)


            temp_re={
                "question_id": question_id,
                "image": img_id,
                "goden answer": answer,
                "question": question,
                "layer": layer,
                "top_k_word":top_k_word,
                "top_k_score":top_k_score,
            }

            if task_name == "ChooseRel" or task_name == "ChooseAttr" or task_name == "ChooseCat":
                true_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(true_option, add_special_tokens=False)[0]]
                false_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(false_option, add_special_tokens=False)[0]]

                true_option_InitialsUpperCase = true_option.capitalize()
                false_option_InitialsUpperCase = false_option.capitalize()

                true_InitialsUpperCase_score_first = scores_first_generated_token[
                    tokenizer.encode(true_option_InitialsUpperCase, add_special_tokens=False)[0]]
                false_InitialsUpperCase_score_first = scores_first_generated_token[
                    tokenizer.encode(false_option_InitialsUpperCase, add_special_tokens=False)[0]]
                temp_re.update({
                    "Noncapitalized Answer": true_LowerCase_score_first*100.0,
                    "Noncapitalized False Option": false_LowerCase_score_first*100.0,
                    "Capitalized Answer": true_InitialsUpperCase_score_first*100.0,
                    "Capitalized False Option": false_InitialsUpperCase_score_first*100.0,
                })
            else:
                answer_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(answer, add_special_tokens=False)[0]]
                answer_InitialsUpperCase = answer.capitalize()
                answer_InitialsUpperCase_score_first = scores_first_generated_token[tokenizer.encode(answer_InitialsUpperCase, add_special_tokens=False)[0]]
                temp_re.update({
                    "Noncapitalized Answer": answer_LowerCase_score_first*100.0,
                    "Capitalized Answer": answer_InitialsUpperCase_score_first*100.0,
                })


            records.append(temp_re)

    tmp = pd.DataFrame.from_records(records)

    save_name = ""
    model_name = model_name.replace('-', '_').replace('.', '_')
    os.makedirs(f"output/last_position_answer_probs/{model_name}/{task_name}/val/", exist_ok=True)
    tmp.to_csv(f'output/last_position_answer_probs/{model_name}/{task_name}/val/{args.refined_dataset.split("/")[-1].split(".csv")[0]}{save_name}.csv', index=False)

    # Plot the results
    save_name += "_" + model_name
    save_path=f'output/last_position_answer_probs/{model_name}/{task_name}/val/{args.refined_dataset.split("/")[-1].split(".csv")[0]}{save_name}_first.pdf'

    if task_name == "ChooseRel" or task_name == "ChooseAttr" or task_name == "ChooseCat":
        measures = [
            "Noncapitalized Answer",
            "Capitalized Answer",
            "Noncapitalized False Option",
            "Capitalized False Option"
        ]
    else:
        measures = [
            "Noncapitalized Answer",
            "Capitalized Answer",
        ]

    generate_plot_attrscore(tmp, save_path, x="layer", ys=measures, layer_num=model.config.num_hidden_layers)









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument('--refined_dataset', default="", type=str, help="refined dataset")

    parser.add_argument("--only_read_cache",action='store_true', default=False)
    parser.add_argument("--only_cache",action='store_true', default=False)



    args = parser.parse_args()

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    main(args)




