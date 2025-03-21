import pdb

import torch
import os
import matplotlib.pyplot as plt
from utils import make_inputs, decode_tokens, predict_from_input
import functools
import time


from types import MethodType


def _split_heads(tensor, num_heads, attn_head_size):
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(1, 0, 2)  # (head, seq_length, head_features)

def _merge_heads(tensor, model):
    num_heads = model.config.n_head
    attn_head_size = model.config.n_embd // model.config.n_head

    tensor = tensor.permute(1, 0, 2).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)


def set_block_attn_add_hooks_llava(model, values_per_layer, coef_value=0, only_knockout_question=None, question_range=[]):
    def change_values(values, coef_val, only_knockout_question, question_range):
        def hook(module, input, output):
            if only_knockout_question:
                assert output.shape[-1] == len(values)
                output[:, question_range, :] = coef_val
            else:
                output[:, :, values] = coef_val
        return hook

    hooks = []
    for layer, values in values_per_layer.items():
        hooks.append(model.model.layers[layer].self_attn.o_proj.register_forward_hook(change_values(values, coef_value, only_knockout_question, question_range)))
    return hooks

def set_block_attn_hooks_llava(model, from_to_index_per_layer, opposite=False, block_desc=None):
    """
    Only works on llava
    """
    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_, block_desc_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):

            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            num_tokens = kwargs["position_ids"][0][-1].item()+1
            q_length = kwargs["hidden_states"][0].size(0)

            if q_length==1:
                if block_desc_.split("->")[-1]=="Last":
                    from_to_index=[(0, t) for _, t in from_to_index_]
                else:
                    from_to_index = []

            else:
                from_to_index=from_to_index_

            if opposite_:
                if q_length == 1:
                    attn_mask = torch.zeros((q_length, num_tokens), dtype=torch.uint8)
                else:
                    attn_mask = torch.tril(torch.zeros((q_length, num_tokens), dtype=torch.uint8))

                if from_to_index !=[]:
                    rows, cols = zip(*from_to_index)
                    attn_mask[rows, cols] = 1
            else:
                if q_length == 1:
                    attn_mask = torch.ones((q_length, num_tokens), dtype=torch.uint8)
                else:
                    attn_mask = torch.tril(torch.ones((q_length, num_tokens), dtype=torch.uint8)) # set the upper triangular part of a matrix (the part above the main diagonal) to zero.

                if from_to_index !=[]:
                    rows, cols = zip(*from_to_index)
                    attn_mask[rows, cols] = 0


            attn_mask = attn_mask.repeat(1, 1, 1, 1)

            attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            attn_mask = attn_mask.to(model_.device)
            new_kwargs["attention_mask"] = attn_mask
            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn

    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = model.model.layers[i].self_attn.forward
        model.model.layers[i].self_attn.forward = wrap_attn_forward(model.model.layers[i].self_attn.forward,
                                                                model, from_to_index_per_layer[i], opposite, block_desc)
        hooks.append((i, hook))

    return hooks



def set_get_attn_proj_hooks(model, tok_index):
    """
    Only works on GPT2
    """
    for attr in ["projs_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_projection(name, E):
        def hook(module, input, output):
            attn_out = output[0][:, tok_index]
            probs, preds = torch.max(
                torch.softmax(attn_out.matmul(E.T), dim=-1),
                dim=-1
            )
            model.projs_[f"{name}_probs"] = probs.cpu().numpy()
            model.projs_[f"{name}_preds"] = preds.cpu().numpy()

        return hook

    E = model.get_input_embeddings().weight.detach()
    hooks = []
    for i in range(model.config.n_layer):
        hooks.append(model.transformer.h[i].attn.register_forward_hook(get_projection(f"attn_proj_{i}", E)))

    return hooks


def set_block_mlp_hooks(model, values_per_layer, coef_value=0):
    def change_values(values, coef_val):
        def hook(module, input, output):
            output[:, :, values] = coef_val

        return hook

    hooks = []
    for layer in range(model.config.n_layer):
        if layer in values_per_layer:
            values = values_per_layer[layer]
        else:
            values = []
        hooks.append(model.transformer.h[layer].mlp.c_fc.register_forward_hook(
            change_values(values, coef_value)
        ))

    return hooks

def set_block_mlp_hooks_llava(model, values_per_layer, coef_value=0,only_knockout_question=None, question_range=[]):
    def change_values(values, coef_val, only_knockout_question, question_range):
        def hook(module, input, output):
            if only_knockout_question:
                assert output.shape[-1] == len(values)
                output[:, question_range, :] = coef_val
            else:
                output[:, :, values] = coef_val
        return hook

    hooks = []
    for layer, values in values_per_layer.items():
        hooks.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(change_values(values, coef_value, only_knockout_question, question_range)))
    return hooks

def set_proj_hooks(model):
    for attr in ["projs_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_projection(name, E):
        def hook(module, input, output):
            num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
            if name == f"layer_residual_{final_layer}":
                hs = output
            else:
                hs = input[0]
            probs, preds = torch.max(
                torch.softmax(hs.matmul(E.T), dim=-1),
                dim=-1
            )
            model.projs_[f"{name}_preds"] = preds.cpu().numpy()
            model.projs_[f"{name}_probs"] = probs.cpu().numpy()

        return hook

    E = model.get_input_embeddings().weight.detach()
    final_layer = model.config.n_layer - 1

    hooks = []
    for i in range(model.config.n_layer - 1):
        hooks.append(model.transformer.h[i].register_forward_hook(
            get_projection(f"layer_residual_{i}", E)
        ))
    hooks.append(model.transformer.ln_f.register_forward_hook(
        get_projection(f"layer_residual_{final_layer}", E)
    ))

    return hooks


def set_hs_patch_hooks(model, hs_patch_config, patch_input=False):
    def patch_hs(name, position_hs, patch_input):

        def pre_hook(module, input):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                input[0][0, position_] = hs_

        def post_hook(module, input, output):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                output[0][0, position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for i in hs_patch_config:
        if patch_input:
            hooks.append(model.transformer.h[i].register_forward_pre_hook(
                patch_hs(f"patch_hs_{i}", hs_patch_config[i], patch_input)
            ))
        else:
            hooks.append(model.transformer.h[i].register_forward_hook(
                patch_hs(f"patch_hs_{i}", hs_patch_config[i], patch_input)
            ))

    return hooks


# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def remove_wrapper_llava(model, hooks):
    for i, hook in hooks:
        model.model.layers[i].self_attn.forward =hook


def trace_with_attn_block_llava(
        model,
        inp,
        from_to_index_per_layer,  # A list of (source index, target index) to block
        first_answer_token_id,
        block_desc,
        model_name
):
    with torch.inference_mode():
        # set hooks
        block_attn_hooks = set_block_attn_hooks_llava(model, from_to_index_per_layer, block_desc=block_desc)

        # get prediction
        output_details = model.generate(**inp)
        answer_token_id = output_details['sequences']  # tensor([[   198,   9619, 128009]], device='cuda:0')  ['\nFull<|eot_id|>']

        logits_first_answer_token = output_details['scores'][0]
        # remove hooks
        remove_wrapper_llava(model, block_attn_hooks)

    [base_score_first] = torch.softmax(logits_first_answer_token, dim=-1)[0][first_answer_token_id]  # (1,1)

    return base_score_first





import math
import torch
from torch.nn import functional as F


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos



def forward_RESAMPLE(self, x, attn_mask=None):
    pos_embed = get_abs_pos(self.pos_embed, x.size(1))

    x = self.kv_proj(x)
    x = self.ln_kv(x).permute(1, 0, 2)

    N = x.shape[1]
    q = self.ln_q(self.query)
    out = self.attn(
        self._repeat(q, N) + self.pos_embed.unsqueeze(1),
        x + pos_embed.unsqueeze(1),
        x,
        attn_mask=attn_mask,
        )
    self.atten_ave_weight_resample=out[1] #[1,256,1024]
    return out[0].permute(1, 0, 2)


def trace_with_proj(model, inp):
    with torch.no_grad():
        # set hooks
        hooks = set_proj_hooks(model)

        # get prediction
        answer_t, base_score = [d[0] for d in predict_from_input(model, inp)]

        # remove hooks
        remove_hooks(hooks)

    projs = model.projs_

    return answer_t, base_score, projs

