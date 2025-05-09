#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-29 12:57:11
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-30 08:54:08
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx


def get_onnx_opset_versions() -> list[int]:
    """
    return opset version list (>=7)

    .. note::
        torch.onnx.export only supports opset version >= 7

    :return: _description_
    :rtype: list[int]
    """
    onnx_opset_version_set: set[int] = set()
    onnx_version_table = onnx.helper.VERSION_TABLE
    for onnx_version in onnx_version_table:
        onnx_opset_version = onnx_version[2]
        if onnx_opset_version < 7:
            continue
        onnx_opset_version_set.add(onnx_opset_version)

    onnx_opset_versions: list[int] = sorted(list(onnx_opset_version_set))
    return onnx_opset_versions


def get_onnx_model_opset_version(model: onnx.ModelProto) -> int:
    for opset_import in model.opset_import:
        if opset_import.domain == "":
            return opset_import.version
        else:
            continue
    raise ValueError("The model does not have a default opset version.")
