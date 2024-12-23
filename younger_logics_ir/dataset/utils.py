#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-13 15:17:27
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-13 15:17:35
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


def get_instance_name_parts(instance_name: str) -> tuple[str, str, str]:
    model_name, instance_name_left = tuple(instance_name.split('--MN_YD_MS--'))
    model_source, onnx_model_filestem = tuple(instance_name_left.split('--MS_YD_ON--'))
    return (model_name, model_source, onnx_model_filestem)


def get_instance_dirname(model_name: str, model_source: str, onnx_model_filestem: str) -> str:
    return model_name + '--MN_YD_MS--' + model_source + '--MS_YD_ON--' + onnx_model_filestem
