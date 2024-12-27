#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-19 22:12:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-27 13:37:40
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger.commons.constants import Constant


class YLIR_OPEN_DATASET_API(Constant):
    def initialize(self) -> None:
        self.ENDPOINT = 'https://younger.yangs.ai/public/'
        self.SERIES_COMPLETE_PATH = 'items/YLIRSeriesComplete'
        self.SERIES_FILTER_PATH = 'items/YLIRSeriesFilter'

YLIROpenDatasetAPI = YLIR_OPEN_DATASET_API()
YLIROpenDatasetAPI.initialize()
YLIROpenDatasetAPI.freeze()


class YLIR_NODE_TYPE(Constant):
    def initialize(self) -> None:
        self.OPERATOR = '__OPERATOR__'
        self.INPUT = '__INPUT__'
        self.OUTPUT = '__OUTPUT__'
        self.OUTER = '__OUTER__'

YLIRNodeType = YLIR_NODE_TYPE()
YLIRNodeType.initialize()
YLIRNodeType.freeze()

class YLIR_ORIGIN_HUB(Constant):
    def initialize(self) -> None:
        self.HUGGINGFACE = 'HuggingFace'
        self.ONNX = 'ONNX'
        self.TORCH = 'Torch'

YLIROriginHub = YLIR_ORIGIN_HUB()
YLIROriginHub.initialize()
YLIROriginHub.freeze()
