#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-17 08:15:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-17 10:20:36
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx

from .onnx2ir import convert_onnx_to_ir
from .core2ir import convert_core_to_ir

from younger_logics_ir.modules import LogicX


def convert(model_handler: onnx.ModelProto | str) -> None:
    """
    :param model: The `model` parameter accepts a static description type of the model logic.
        It accepts the following types:
            1. `onnx.ModelProto` a ONNX format model;
            2. `str` a Human-readable model definition.
    """

    logicx = LogicX
    if isinstance(model_handler, onnx.ModelProto):
        logicx = LogicX(dag=convert_onnx_to_ir(model_handler), src='onnx')

    if isinstance(model_handler, str):
        logicx = LogicX(dag=convert_core_to_ir(model_handler), src='core')

    assert logicx.valid, f'LogicX has not been setup correctly! Validate - LogicX.dag: {logicx.dag_valid}; LogicX.src: {logicx.src_valid}'

    return logicx