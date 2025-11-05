# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np

def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    healthy = [
        '{}', 'healthy {}', 'perfectly healthy {}', 'no tumor in {}', '{} without a tumor',  
        '{} with no tumors', 'normal brain MRI of {}', 'healthy scan of {}', '{} shows no sign of a tumor',
        'no anomalies detected in {}'
    ]

    glioma = [
        '{} has a glioma', '{} shows signs of a glioma', 'glioma found on {}',
        '{} with a visible glioma', 'MRI shows a glioma in {}', '{} affected by a glioma tumor'
    ]

    meningioma = [
        '{} has a meningioma', 'a meningioma visible on {}', 'meningioma found on {}',
        '{} with a meningioma tumor', 'presence of a meningioma in {}', '{} meningioma growth on {}'
    ]

    pituitary = [
        '{} with a pituitary', '{} shows a pituitary', 'visible pituitary on {}',
        '{} has a pituitary', 'pituitary growth visible in {}', 'pituitary tumor detected in {}'
    ]

    prompt_state = [healthy, glioma, meningioma, pituitary]

    prompt_templates = ['an MRI scan of {}', 'MRI showing {}', 'brain MRI scan with {}', 'a medical scan of {}', 'an MRI image showing {}']

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