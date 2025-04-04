import os
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T
import albumentations as A
import open_clip
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from tqdm import tqdm
from PIL import Image

from utils import (
    DatasetHC18,
    DIR_TEST,
    LIST_TEST_PID,
)

class HF_VisualEncoderWithHooks(nn.Module):
    """
    REFERENCES:
    - https://github.com/huggingface/pytorch-image-models/blob/2703d155c88d27bba9a1f465f5489a7947ffc313/timm/models/vision_transformer.py#L414
    """
    def __init__(self, visual_encoder):
        super(HF_VisualEncoderWithHooks, self).__init__()
        self.visual_encoder = visual_encoder
        self.hooks = []
        self.intermediate_outputs = {}

        self.width = self.visual_encoder.transformer.width
        self.grid_size = self.visual_encoder.grid_size
        
        # Register hooks when the class is initialized
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register hooks for the layers specified in self.layers_to_hook.
        """
        n_blocks = len(self.visual_encoder.transformer.resblocks)
        for layer_idx in [n_blocks // i - 1 for i in range(1,5)]:
            layer = self.visual_encoder.transformer.resblocks[layer_idx]
            hook = layer.register_forward_hook(self._get_intermediate_output(f'layer_{layer_idx+1}'))
            self.hooks.append(hook)

    def _get_intermediate_output(self, layer_name):
        """
        Hook function to capture the intermediate output.
        """
        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output
        return hook

    def forward(self, x):
        """
        Perform the forward pass while capturing intermediate outputs.
        """
        # Reset intermediate outputs before forward pass
        self.intermediate_outputs = {}
        
        # Perform the forward pass of the VisionTransformer
        output = self.visual_encoder(x)

        list_keys = sorted(list(self.intermediate_outputs.keys()), key=lambda x: int(x.split('_')[1]))
        intermediate_outputs = [
            self.intermediate_outputs[key].permute(1,0,2)[1:]
            for key in list_keys
        ]
        
        return output, intermediate_outputs

    def remove_hooks(self):
        """
        Remove all hooks after usage.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class TIMM_VisualEncoderWithHooks(nn.Module):
    """
    REFERENCES:
    - https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L434
    """
    def __init__(self, visual_encoder):
        super(TIMM_VisualEncoderWithHooks, self).__init__()
        self.visual_encoder = visual_encoder
        self.hooks = []
        self.intermediate_outputs = {}

        self.width = self.visual_encoder.trunk.embed_dim
        self.grid_size = self.visual_encoder.trunk.patch_embed.grid_size
        
        # Register hooks when the class is initialized
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register hooks for the layers specified in self.layers_to_hook.
        """
        n_blocks = len(self.visual_encoder.trunk.blocks)
        for layer_idx in [n_blocks // i - 1 for i in range(1,5)]:
            layer = self.visual_encoder.trunk.blocks[layer_idx]
            hook = layer.register_forward_hook(self._get_intermediate_output(f'layer_{layer_idx+1}'))
            self.hooks.append(hook)

    def _get_intermediate_output(self, layer_name):
        """
        Hook function to capture the intermediate output.
        """
        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output
        return hook

    def forward(self, x):
        """
        Perform the forward pass while capturing intermediate outputs.
        """
        # Reset intermediate outputs before forward pass
        self.intermediate_outputs = {}
        
        # Perform the forward pass of the VisionTransformer
        output = self.visual_encoder(x)

        list_keys = sorted(list(self.intermediate_outputs.keys()), key=lambda x: int(x.split('_')[1]))
        intermediate_outputs = [
            self.intermediate_outputs[key].permute(1,0,2)[1:]
            for key in list_keys
        ]
        
        return output, intermediate_outputs

    def remove_hooks(self):
        """
        Remove all hooks after usage.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class EncoderWrapper(nn.Module):
    def __init__(self, visual_encoder):
        super(EncoderWrapper, self).__init__()

        if isinstance(visual_encoder, open_clip.transformer.VisionTransformer):
            self.transformer = HF_VisualEncoderWithHooks(visual_encoder)
        elif isinstance(visual_encoder, open_clip.timm_model.TimmModel):
            self.transformer = TIMM_VisualEncoderWithHooks(visual_encoder)
    
    def forward(self, x):
        with torch.no_grad():
            z = self.transformer(x)
        
            z0, z3, z6, z9, z12 = x, *z[1]
            z3 = z3.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            z6 = z6.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            z9 = z9.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            z12 = z12.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            
            z3 = F.interpolate(z3, size=(14, 14), mode='bilinear', align_corners=False)
            z6 = F.interpolate(z6, size=(14, 14), mode='bilinear', align_corners=False)
            z9 = F.interpolate(z9, size=(14, 14), mode='bilinear', align_corners=False)
            z12 = F.interpolate(z12, size=(14, 14), mode='bilinear', align_corners=False)
        
        return {
            'z3': z3, 'z6': z6,
            'z9': z9, 'z12': z12,
        }

def get_list_embeddings(model, dl):
    list_z3 = []
    list_z6 = []
    list_z9 = []
    list_z12 = []

    for x in tqdm(dl):
        x = x[0].to('cuda')
        with torch.no_grad():
            z = model(x)
        list_z3.append(z['z3'].detach().cpu())
        list_z6.append(z['z6'].detach().cpu())
        list_z9.append(z['z9'].detach().cpu())
        list_z12.append(z['z12'].detach().cpu())

    z3 = torch.cat(list_z3, dim=0)
    z6 = torch.cat(list_z6, dim=0)
    z9 = torch.cat(list_z9, dim=0)
    z12 = torch.cat(list_z12, dim=0)

    return [
        [a.numpy(), b.numpy(), c.numpy(), d.numpy()]
        for a, b, c, d in zip(z3, z6, z9, z12)
    ]

def get_dict_embeddings(
        model, ds,
        batch_size=8, num_workers=4,
    ):
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    list_embeddings = get_list_embeddings(model, dl)

    dict_embeddings = {}
    for filepath, embedding in tqdm(zip(ds.data, list_embeddings)):
        filename = os.path.basename(filepath).split('.')[0]
        dict_embeddings[filename] = embedding
    
    return dict_embeddings

def get_test_embeddings(
        model, preprocess_img, preprocess_mask,
        batch_size=8, num_workers=4,
    ):
    ds = DatasetHC18(
        DIR_TEST,
        LIST_TEST_PID,
        preprocess_img,
        preprocess_mask,
    )

    dict_embeddings = get_dict_embeddings(model, ds, batch_size, num_workers)

    return dict_embeddings

def get_train_val_embeddings(
        model, root_train, root_val, train_val_pid,
        preprocess_img, preprocess_mask,
        batch_size=8, num_workers=4,
    ):
    ds_train = DatasetHC18(
        root_train,
        train_val_pid[0],
        preprocess_img,
        preprocess_mask,
    )
    ds_val = DatasetHC18(
        root_val,
        train_val_pid[1],
        preprocess_img,
        preprocess_mask,
    )

    dict_train_embeddings = get_dict_embeddings(model, ds_train, batch_size, num_workers)
    dict_val_embeddings   = get_dict_embeddings(model, ds_val,   batch_size, num_workers)

    return dict_train_embeddings, dict_val_embeddings