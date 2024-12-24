#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-16 10:31:35
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-23 16:51:23
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Any


class Origin(object):
    def __init__(self, hub: str, owner: str, name: str, like: int, download: int):
        self._hub = hub
        self._owner = owner
        self._name = name
        self._like = like
        self._download = download


class Evaluation(object):
    def __init__(self, task: str, dataset: str, results: dict[str, Any] | None = None):
        self._task = task
        self._dataset = dataset
        self._results = results or dict()

    def clean_results(self) -> None:
        self._results.clear()

    def insert_result(self, metric_name: str, metric_value: Any) -> Any:
        self._results[metric_name] = metric_value
        return self._results[metric_name]

    def delete_result(self, metric_name: str) -> Any:
        metric_value = self._results.pop(metric_name)
        return metric_value

    def update_result(self, metric_name: str, metric_value: Any) -> None:
        self._results[metric_name] = metric_value
        return self._results[metric_name]


class Label(object):
    def __init__(self, origins: list[Origin] | None = None, evaluations: list[Evaluation] | None = None):
        self._origins = origins or list()
        self._evaluations = evaluations or list()

    def __hash__(self) -> int:
        hash()

    def __eq__(self) -> bool:
        pass

    @classmethod
    def loads(cls, txt: str) -> 'Label':
        pass

    @classmethod
    def saves(cls, lbl: 'Label') -> str:
        pass
