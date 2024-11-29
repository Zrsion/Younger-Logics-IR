#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-27 15:26:33
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger_logics_ir.utils.detectors.langs import detect_natural_langs, detect_program_langs
from younger_logics_ir.utils.detectors.tasks import detect_task
from younger_logics_ir.utils.detectors.datasets import detect_dataset_name, detect_dataset_split
from younger_logics_ir.utils.detectors.metrics import detect_metric_name, normalize_metric_value