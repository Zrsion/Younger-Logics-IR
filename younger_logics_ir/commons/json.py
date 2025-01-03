#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-03 21:02:51
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-03 22:53:16
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import json
import base64


BYTESHead = 'YLIR-BYTES-'


class YLIRJSONEncoder(json.JSONEncoder):
    def default(self, field):
        if isinstance(field, bytes):
            return f'{BYTESHead}{base64.b64encode(field).decode("utf-8")}'
        return json.JSONEncoder.default(self, field)


class YLIRJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(
            object_hook=self.object_hook,
            *args,
            **kwargs
        )

    def object_hook(self, obj):
        for key, value in obj.items():
            if isinstance(value, str):
                obj[key] = self.decode_bytes(value)
            elif isinstance(value, list):
                obj[key] = [self.decode_bytes(item) if isinstance(item, str) else item for item in value]
            elif isinstance(value, dict):
                obj[key] = self.object_hook(value)
        return obj

    def decode_bytes(self, obj):
        if isinstance(obj, str) and obj.startswith(BYTESHead):
            try:
                obj = base64.b64decode(obj[len(BYTESHead):].encode('utf-8'))
            except (base64.binascii.Error, TypeError):
                return obj
        return obj
