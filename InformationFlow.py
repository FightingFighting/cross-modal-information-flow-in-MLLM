
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


from PIL import Image, ImageDraw


import argparse
import os
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerateOutput
import requests
import copy

from utils import prepare_image_patch_bbx,create_mask_with_bbox,show_original_image,show_transferred_maskandimage, generate_plot


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.image_processor_mask = copy.deepcopy(image_processor)
        self.model_name = get_model_name_from_path(self.model_config._name_or_path)
        self.task_name = task_name
        self.conv_mode = conv_mode

        if self.model_name == "llama3-llava-next-8b" or self.model_name == "llava-v1.6-vicuna-7b" or self.model_name == "llava-v1.5-7b" or self.model_name == "llava-v1.5-13b":
            self.image_processor_mask.do_normalize=False
            self.image_processor_mask.do_rescale=False
        elif self.model_name == "llava-next-qwen-32b":
            self.image_processor_mask.image_mean = (0, 0, 0)
            self.image_processor_mask.image_std = (1, 1, 1)
            self.image_processor_mask.rescale_factor = 1

    def __getitem__(self, index):

        line = self.questions[index]
        question = line["question"]
        question = question + " \nAnswer the question using a single word or phrase."
        image_file = str(line["img_id"]) + ".jpg" if ".jpg" not in str(line["img_id"]) else str(line["img_id"])


        qs = DEFAULT_IMAGE_TOKEN + "\n" + question  #

        conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if self.model_name == "llama3-llava-next-8b":
            prompt+=" \n"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")

        image_tensor = process_images([image], self.image_processor, self.model_config)
        image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]

        if self.task_name == "CompareAttr" or self.task_name == "ChooseRel" or self.task_name == "LogicalObj":
            bounding_boxes=[]
            bounding_boxes.append((int(line[f'object1 x']), int(line[f'object1 y']), int(line[f'object1 x'])+int(line[f'object1 w']), int(line[f'object1 y'])+int(line[f'object1 h'])))
            if line[f'object2 x'] !="-":
                bounding_boxes.append((int(line[f'object2 x']), int(line[f'object2 y']), int(line[f'object2 x'])+int(line[f'object2 w']), int(line[f'object2 y'])+int(line[f'object2 h'])))
        elif self.task_name=="ChooseAttr" or self.task_name=="ChooseCat" or self.task_name=="QueryAttr":
            bounding_boxes = [(int(line['central object x']), int(line['central object y']), int(line['central object x'])+int(line['central object w']), int(line['central object y'])+int(line['central object h']))]
        else:
            bounding_boxes = None

        if bounding_boxes !=None:
            mask = create_mask_with_bbox(image, bounding_boxes)
            mask_tensor = process_images([mask], self.image_processor_mask, self.model_config)
            mask_tensor = [_image.to(dtype=torch.float16) for _image in mask_tensor]
        else:
            mask_tensor=None

        # show_original_image(image, bounding_boxes, self.model_name.replace('-', '_').replace('.', '_'), save_name=str(line["img_id"]), question = line["question"], answer=line["answer"])
        # if mask_tensor[0].ndim==3:
        #     for ind, (ma, img) in enumerate(zip(mask_tensor, image_tensor)):
        #         show_transferred_maskandimage(ma,img, ind, self.model_name.replace('-', '_').replace('.', '_'), save_name=str(line["img_id"]))
        # else:
        #     for ind, (ma, img) in enumerate(zip(mask_tensor[0], image_tensor[0])):
        #         show_transferred_maskandimage(ma,img, ind, self.model_name.replace('-', '_').replace('.', '_'), save_name=str(line["img_id"]))

        image_sizes = [image.size]
        return input_ids, image_tensor, image_sizes, prompt, mask_tensor




    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompts, mask_tensors = zip(*batch)

    input_ids = input_ids[0]
    image_tensors = image_tensors[0]
    image_sizes=image_sizes[0]
    mask_tensors=mask_tensors[0]
    return input_ids, image_tensors, image_sizes, prompts,mask_tensors



def create_data_loader(questions, image_folder, batch_size, num_workers, tokenizer, image_processor, model_config, task_name, conv_mode):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def find_token_range(tokenizer, token_array, substring, model_name):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = tokenizer.convert_ids_to_tokens(token_array)


  if model_name == model_name == "llava-v1.6-vicuna-7b" or model_name == "llava-v1.5-7b" or model_name == "llava-v1.5-13b":
      whole_string = "".join(toks).replace("▁", " ")
  elif model_name == "llama3-llava-next-8b" or "llava-next-qwen-32b":
    whole_string = "".join(toks).replace("Ġ"," ").replace("Ċ","\n")


  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)



@torch.no_grad()
def generate_llava(
    self,
    mask=None, #[5, 3, 336, 336]
    args=None,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")


    if images is not None:
        if args.certain_part_image:
            (inputs_, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            patched_mask = self.prepare_image_patch_bbx(mask, image_sizes=image_sizes) #[2352, 14*14, 3]   patch_size:14
            patched_mask = np.array(patched_mask[0].cpu())
            target_object = np.array([255, 0, 0], dtype=np.uint8) #red
            match_object = np.all(patched_mask == target_object, axis=-1)
            objects_indices = np.where(np.any(match_object, axis=1))[0]
            target_pad = np.array([-1, -1, -1], dtype=np.int8) #pad
            match_pad = np.all(patched_mask == target_pad, axis=-1)
            pad_indices = np.where(np.any(match_pad, axis=1))[0]
            original_patch_number = (mask[0].size(-1)//self.get_vision_tower().config.patch_size)**2
            original_patch_indices = list(range(patched_mask.shape[0]))[0:original_patch_number]
            hd_patch_indice = list(range(patched_mask.shape[0]))[original_patch_number:]
            objects_indices_in_hd =objects_indices[objects_indices>=original_patch_number]
            return patched_mask, objects_indices, pad_indices,original_patch_indices,hd_patch_indice,objects_indices_in_hd, inputs_embeds.shape, super(self.__class__, self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        else:
            (inputs_, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            return inputs_embeds.shape, super(self.__class__, self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)







def run_original(model, inps, tokenizer, model_name, mask_tensor=None, args=None):
    with torch.inference_mode():
        model.old_generate= model.generate
        model.generate =  MethodType(generate_llava, model)
        if args.certain_part_image:
            patched_mask, objects_indices, pad_indices,original_patch_indices,hd_patch_indice, objects_indices_in_hd, inputs_embeds_shape, output_details = model.generate(mask=mask_tensor, args=args, **inps)
        else:
            inputs_embeds_shape, output_details = model.generate(args=args,**inps)
        model.generate = model.old_generate


    answer_token_id = output_details['sequences']

    first_answer_token_id = answer_token_id[:, 0]
    logits_first_answer_token = output_details['scores'][0]

    [base_score_first] = torch.softmax(logits_first_answer_token, dim=-1)[0][first_answer_token_id]  # (1,1)
    base_score_first = base_score_first.item()

    predicted_answer = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip().lower()

    if args.certain_part_image:
        return base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape, objects_indices, pad_indices, original_patch_indices,hd_patch_indice, objects_indices_in_hd, patched_mask
    else:
        return base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape



def blockdesc2range(des, dataset_dict, question_id, input_ids, inputs_embeds_shape, tokenizer, model_name):
    if des=="Last":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        ntoks = input_ids.shape[1] + image_dim - 1
        source_ = ntoks - 1
        return [source_]
    if des=="Question":
        question = dataset_dict[question_id]["question"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        question_range = find_token_range(tokenizer, input_ids_noim[1], question, model_name)
        question_range = [x for x in range(question_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                           question_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return question_range
    if des=="Image":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_range = [x for x in range(torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0],
                                        torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0] + image_dim)]
        return image_range
    if des=="True Option":
        true_option = dataset_dict[question_id]["true option"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [ input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        true_option_range = find_token_range(tokenizer, input_ids_noim[1], true_option, model_name)
        true_option_range = [x for x in range(true_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                              true_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return true_option_range
    if des=="False Option":
        false_option = dataset_dict[question_id]["false option"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
            input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        false_option_range = find_token_range(tokenizer, input_ids_noim[1], false_option, model_name)
        false_option_range = [x for x in range(false_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                               false_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return false_option_range
    if des=="Central Object":
        central_object = dataset_dict[question_id]["central object name"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
            input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        central_object_range = find_token_range(tokenizer, input_ids_noim[1], central_object, model_name)
        central_object_range = [x for x in range(central_object_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                                 central_object_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return central_object_range
    if des=="Question without Options":
        true_option = dataset_dict[question_id]["true option"]
        false_option = dataset_dict[question_id]["false option"]
        question = dataset_dict[question_id]["question"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
            input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        true_option_range = find_token_range(tokenizer, input_ids_noim[1], true_option, model_name)
        false_option_range = find_token_range(tokenizer, input_ids_noim[1], false_option, model_name)
        question_range = find_token_range(tokenizer, input_ids_noim[1], question, model_name)
        true_option_range = [x for x in range(true_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                              true_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        false_option_range = [x for x in range(false_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                               false_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        question_range = [x for x in range(question_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                           question_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        question_withoutOptions_range = [item for item in question_range if
                                         item not in true_option_range + false_option_range]
        return question_withoutOptions_range


def blockdesc2range_patches(des, input_ids, inputs_embeds_shape, central_object_patch_indices, pad_patch_indices, hd_patch_indice, objects_indices_in_hd, original_patch_indices):
    if des=="Image Without Central Object":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        other_indices_without_central_object = list(set(range(image_dim)) - set(central_object_patch_indices) - set(pad_patch_indices))
        image_without_central_object_range = (np.array(other_indices_without_central_object) + image_index).tolist()
        return image_without_central_object_range
    if des=="Image Central Object":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        image_central_object_range = (np.array(central_object_patch_indices) + image_index).tolist()
        return image_central_object_range
    if des=="Image Pad":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        image_pad_range = (np.array(pad_patch_indices) + image_index).tolist()
        return image_pad_range
    if des=="Image Without Central Object with pad":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        other_indices_without_central_object = list(set(range(image_dim)) - set(central_object_patch_indices) - set(pad_patch_indices))
        image_pad_range = (np.array(pad_patch_indices) + image_index).tolist()
        image_without_central_object_range = (np.array(other_indices_without_central_object) + image_index).tolist()
        return image_without_central_object_range + image_pad_range
    if des=="Image Original Patch":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        original_patch_range = (np.array(original_patch_indices) + image_index).tolist()
        return original_patch_range
    if des=="Image HD Patch Indice":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        hd_patch_indice_range = (np.array(hd_patch_indice) + image_index).tolist()
        return hd_patch_indice_range
    if des=="Image Central Object in HD Patch Indice":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        objects_indices_in_hd_range = (np.array(objects_indices_in_hd) + image_index).tolist()
        return objects_indices_in_hd_range
    if des=="Image HD Patch Without Central Object Indice":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        other_indices_without_central_object_in_hd = list(set(hd_patch_indice) - set(objects_indices_in_hd))
        other_indices_without_central_object_in_hd_range = ( np.array(other_indices_without_central_object_in_hd) + image_index).tolist()
        return other_indices_without_central_object_in_hd_range





# Information flow analysis
def InforFlowAna(args):


    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base,model_name,device_map="auto",attn_implementation=None)
    model.prepare_image_patch_bbx=MethodType(prepare_image_patch_bbx, model)
    model.eval()

    if args.noHD_noPad:
        model.config.image_aspect_ratio="pad"  #HD: anyres
        model.config.mm_patch_merge_type='spatial'  # pad: 'spatial_unpad

    #dataset
    #predict correct and filter
    task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
    df = pd.read_csv(args.refined_dataset, dtype={"question_id":str}).fillna('')
    dataset_dict = df.set_index('question_id').T.to_dict('dict')
    questions = [ {**detail, "q_id":qu_id} for qu_id, detail in dataset_dict.items()]
    data_loader = create_data_loader(questions, args.image_folder,  args.batch_size, args.num_workers, tokenizer,  image_processor, model.config, task_name, args.conv_mode)



    # Run attention knockouts
    results = []
    index=0
    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        if index==20:
            break
        question_id = line["q_id"]
        img_id= str(line["img_id"]) + ".jpg"

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]
        mask_tensor = [ma.to(device='cuda') for ma in mask_tensor]

        inps={
            "inputs":input_ids,
            "images":image_tensor,
            "image_sizes":original_image_sizes,
            "do_sample":True if args.temperature > 0 else False,
            "temperature":args.temperature,
            "top_p":args.top_p,
            "num_beams":args.num_beams,
            "max_new_tokens" : args.max_new_tokens,
            "use_cache" : True,
            "return_dict_in_generate" : True,
            "output_scores" : True,
            "pad_token_id": tokenizer.eos_token_id

        }

        question = dataset_dict[question_id]["question"]
        answer = dataset_dict[question_id]["answer"]


        if args.certain_part_image:
            base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape,central_object_patch_indices, pad_patch_indices,original_patch_indices,hd_patch_indice,  objects_indices_in_hd, patched_mask = run_original(model, inps, tokenizer, model_name, mask_tensor, args=args)
        else:
            base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape = run_original(model, inps, tokenizer, model_name, args=args)

        if answer!=predicted_answer:
            continue
        else:
            index += 1
            print("Finish samples:", index)


        #get range
        block_descs_split = args.block_description.split("->")
        if args.certain_part_image:
            range1 = blockdesc2range_patches(block_descs_split[0], input_ids, inputs_embeds_shape,central_object_patch_indices, pad_patch_indices, hd_patch_indice,objects_indices_in_hd, original_patch_indices)
        else:
            range1 = blockdesc2range(block_descs_split[0], dataset_dict, question_id, input_ids, inputs_embeds_shape,tokenizer, model_name)
        range2 = blockdesc2range(block_descs_split[1], dataset_dict, question_id, input_ids, inputs_embeds_shape,tokenizer, model_name)
        block_descs = [([range1, range2], args.block_description)]

        for block_ids, block_desc in block_descs:

            temp2 = [(stok1, stok0) for stok0 in block_ids[0] for stok1 in block_ids[1]]

            for layer in range(model.config.num_hidden_layers):
                layerlist = [
                    l for l in range(
                        max(0, layer - args.window // 2), min(model.config.num_hidden_layers, layer - (-args.window // 2))
                    )
                ]
                block_config = {
                    l:copy.deepcopy(temp2)
                    for l in layerlist
                }

                inps["max_new_tokens"] = 1
                new_score_first = trace_with_attn_block_llava(
                    model, inps, block_config, first_answer_token_id, block_desc, model_name
                )
                new_score_first = new_score_first.cpu().item()

                re={
                    "question_id": question_id,
                    "image": img_id,
                    "goden answer": answer,
                    "predicted answer": predicted_answer,
                    "question": question,
                    "block_desc": block_desc,
                    "layer": layer,
                    "base_score_first": base_score_first,
                    "new_score_first": new_score_first,
                    "relative diff first": (new_score_first - base_score_first) * 100.0 / base_score_first,
                }
                results.append(re)


    save_name = "_".join([des[1].replace(" ", "_").replace("->", "___") for des in block_descs])
    if args.noHD_noPad:
        save_name=save_name+"_noHD_noPad"
    tmp = pd.DataFrame.from_records(results)
    model_name = model_name.replace('-', '_').replace('.', '_')
    os.makedirs(f"output/information_flow/{model_name}/{task_name}/val/{save_name}", exist_ok=True)
    tmp.to_csv(f'output/information_flow/{model_name}/{task_name}/val/{save_name}/{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}.csv', index=False)

    # Plot the results
    save_path=f'output/information_flow/{model_name}/{task_name}/val/{save_name}/{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}_first.pdf'
    generate_plot(tmp, save_path, x="layer", y="relative diff first", hue="block_desc", layers=model.config.num_hidden_layers)



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


    parser.add_argument("--window", type=int, default=9)
    parser.add_argument('--refined_dataset', default="", type=str, help="refined dataset")
    parser.add_argument('--block_description', default=None, type=str, help="block_description")
    parser.add_argument('--certain_part_image', default=False, action="store_true")
    parser.add_argument('--noHD_noPad', default=False, action="store_true", help="noHD_noPad")


    args = parser.parse_args()

    block_descs_split = args.block_description.split("->")
    if "Image " in block_descs_split[0]:
        args.certain_part_image=True

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    InforFlowAna(args)


