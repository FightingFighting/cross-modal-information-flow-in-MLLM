# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re

import torch
import transformers
import seaborn as sns
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import copy


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      ):
    if tokenizer is None:
      assert model_name is not None
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if model is None:
      assert model_name is not None
      model = transformers.AutoModelForCausalLM.from_pretrained(
          model_name, low_cpu_mem_usage=low_cpu_mem_usage,
          torch_dtype=torch_dtype
          )
      set_requires_grad(False, model)
      model.eval().cuda()
    self.tokenizer = tokenizer
    self.model = model
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]

  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)

  whole_string = "".join(toks)
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


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)



def generate_plot(data, save_file,x="layer",y="score_all_objects",hue=None, layers=0):

    sns.set(context="notebook",
            rc={"font.size": 14,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 14.0,
                "ytick.labelsize": 14.0,
                "legend.fontsize": 10.0})
    palette_ = sns.color_palette("Set1") #9 colors
    palette = palette_[2:5] + palette_[5:6] + palette_[7:] + palette_[0:2] + palette_[6:7]
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(4, 4))
    ax = sns.lineplot(data, x=x, y=y,
                      hue=hue,
                      style=hue,
                      dashes=True,
                      palette=palette, linewidth=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("% change in prediction probability")
    ax.set_xlim(0, layers+0.5)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.legend(title='blocked positions',fontsize=8,handlelength=2, handletextpad=0.1)
    plt.savefig(save_file)


def create_mask_with_bbox(image, bboxes):

    mask = Image.new('RGB', image.size, (0, 255, 0))
    draw = ImageDraw.Draw(mask)
    unique_color = (255, 0, 0)
    for bbox in bboxes:
        draw.rectangle(bbox, fill=unique_color)
    return mask


def show_transferred_maskandimage(mask, img, ind, model_name, save_name):
    mask_array = np.array(mask.cpu())

    red_channel = mask_array[0, :, :]
    green_channel = mask_array[1, :, :]
    blue_channel = mask_array[2, :, :]

    unique_indices = np.argwhere((red_channel == 255) & (green_channel == 0) & (blue_channel == 0))

    if len(unique_indices) > 0:
        top_left = unique_indices.min(axis=0)
        bottom_right = unique_indices.max(axis=0)
        new_bounding_box = (top_left[1], top_left[0], bottom_right[1], bottom_right[0])
        print("transfered bounding box:", new_bounding_box)


        mask = Image.fromarray(mask.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rectangle(new_bounding_box, outline="blue", width=3)
        mask.save(f"output/information_flow/{model_name}/Transformed_Mask_{ind}_{save_name}.jpg")

        img = Image.fromarray((img*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        draw_img = ImageDraw.Draw(img)
        draw_img.rectangle(new_bounding_box, outline="blue", width=3)
        img.save(f"output/information_flow/{model_name}/Transformed_Image_{ind}_{save_name}.jpg")

    else:
        print("didnot find the transfered bounding box")

        mask = Image.fromarray(mask.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        mask.save(f"output/information_flow/{model_name}/Transformed_Mask_{ind}_{save_name}.jpg")

        img = Image.fromarray((img*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        img.save(f"output/information_flow/{model_name}/Transformed_Image_{ind}_{save_name}.jpg")



def show_original_image(image, bounding_boxes, model_name, save_name, question, answer):
    image.save(f"output/information_flow/{model_name}/images/original_image_{save_name}.jpg")
    image_with_bbox = copy.deepcopy(image)
    draw = ImageDraw.Draw(image_with_bbox)
    for bounding_box in bounding_boxes:
        draw.rectangle(bounding_box, outline="blue", width=3)

    x = 0
    y = 0
    draw.text((x, y), question+" "+answer, fill=(0, 0, 0))


    image_with_bbox.save(f"output/information_flow/{model_name}/images/image_withbbx_text_{save_name}.jpg")





from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
from llava.model.llava_arch import unpad_image

def prepare_image_patch_bbx(self, images, image_sizes=None):
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)  # [5, 3, 336, 336]
        _, _, image_w, image_h = concat_images.size()
        assert image_w == image_h
        image_size = image_h = image_w
        patch_size = self.get_model().get_vision_tower().config.patch_size
        num_batch, channel, _, _ = concat_images.size()
        concat_images_patches = concat_images.unfold(2, patch_size, patch_size).unfold(3, patch_size,
                                                                                       patch_size)  # [5, 3, 24, 24, 14, 14]
        concat_images_patches = concat_images_patches.reshape(num_batch, channel, image_size // patch_size,
                                                              image_size // patch_size,
                                                              patch_size * patch_size)  # [5, 3, 24, 24, 14*14]
        concat_images_patches = concat_images_patches.contiguous().view(num_batch, channel, -1,
                                                                        patch_size * patch_size)  # [5, 3, 576, 196]
        concat_images_patches = concat_images_patches.permute(0, 2, 3, 1).contiguous()  # [5, 576, 196, 3]

        split_sizes = [image.shape[0] for image in images_list]
        image_patches = torch.split(concat_images_patches, split_sizes)  # [(5, 576, 196, 3)]

        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

        if mm_patch_merge_type.startswith("spatial"):
            new_image_patches = []
            for image_idx, image_pat in enumerate(image_patches):
                if image_pat.shape[0] > 1:  # multi patches and multi images operations
                    # rank0_print("Single-images")
                    base_image_pat = image_pat[0]  # [576, 196, 3]
                    image_pat = image_pat[1:]  # [4, 576, 196, 3]

                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_pat.shape[0]

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx],
                                                                                            self.config.image_grid_pinpoints,
                                                                                            vision_tower_image_size)
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        # image_pat = image_pat.view(num_patch_height, num_patch_width, height, width, -1) #[2,2,24,24,4096]
                        image_pat = image_pat.view(num_patch_height, num_patch_width, height, width,
                                                   patch_size * patch_size * channel)  # [2, 2, 24, 24, 196*3]

                    if "unpad" in mm_patch_merge_type:
                        image_pat = image_pat.permute(4, 0, 2, 1, 3).contiguous()  # [196*3, 2, 24, 2, 24]
                        image_pat = image_pat.flatten(1, 2).flatten(2, 3)  # [196*3, 48, 48]
                        image_pat = unpad_image(image_pat, image_sizes[image_idx])  # ([196*3, 48, 36])

                        image_pat = torch.cat((image_pat,
                                               (torch.ones([image_pat.size(0)]) * (-1))[:, None, None].expand(
                                                   *image_pat.shape[:-1], 1).to(image_pat.device)),
                                              dim=-1)  # ([196*3, 48, 37])
                        image_pat = image_pat.flatten(1, 2)  # [196*3, 1776]
                        image_pat = image_pat.permute(1, 0)  # [1776, 196*3]
                        image_pat = image_pat.view(-1, patch_size * patch_size, channel)
                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_pat = torch.cat((base_image_pat, image_pat), dim=0)  # [2352, 196, 3]  576+1776
                else:
                    image_pat = image_pat[0]

                new_image_patches.append(image_pat)
        else:
            new_image_patches = [image_patches[0].squeeze(0)]
    image_patches = new_image_patches

    return image_patches

