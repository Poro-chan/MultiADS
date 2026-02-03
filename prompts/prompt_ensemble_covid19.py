# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np

def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    good = [
        '{}', 'healthy {}', 'perfectly healthy {}', 'no disease in {}', '{} without a disease',  
        '{} with no disease', 'normal photo of {}', 'healthy picture of {}', '{} shows no sign of a disease',
        'no anomalies detected in {}'
    ]

    covid = [
        '{} has COVID', '{} shows signs of COVID', 'COVID found on {}',
        '{} with visible COVID', 'shows COVID in {}', '{} affected by COVID'
    ]

    lung_opacity = [
        '{} has lung opacity', 'a lung opacity visible on {}', 'lung opacity found on {}',
        '{} with lung opacity', 'presence of a lung opacity in {}', 'lung opacity on {}'
    ]

    viral_pneumonia = [
        '{} with viral pneumonia', '{} shows viral pneumonia', 'visible viral pneumonia on {}',
        '{} has viral pneumonia', 'viral pneumonia visible on {}', 'viral pneumonia detected in {}'
    ]

    prompt_state = [good, covid, lung_opacity, viral_pneumonia]

    prompt_templates = ['a photo of a {}', 'photo showing {}', 'photo with {}', 'a medical photo of {}', 'an image showing {}']

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts