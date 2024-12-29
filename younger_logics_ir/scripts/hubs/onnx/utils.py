#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:19
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-29 20:34:10
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from onnx import hub
from typing import Any


def get_onnx_hub_model_infos() -> list[dict[str, Any]]:
    model_infos = dict()
    for model in sorted(hub.list_models(), key=lambda x: x.metadata['model_bytes']):
        model_id = model.model
        model_info = model_infos.get(model_id, list())
        model_info.append(
            dict(
                tags = list(model.tags),
                meta = model.metadata,
                path = model.model_path,
                hash = model.model_sha,
                opset = model.opset,
                raw = model.raw_model_info,
            )
        )
        model_infos[model_id] = model_info
    model_infos = [dict(id=model_id, variations=model_info) for model_id, model_info in model_infos.items()]
    return model_infos


def get_onnx_hub_model_ids() -> list[str]:
    model_infos = get_onnx_hub_model_infos()
    model_ids = [model_info['id'] for model_info in model_infos]
    return model_ids
