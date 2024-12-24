#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-16 10:31:35
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-24 15:32:12
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Any

from younger.commons.io import loads_json, saves_json


class Origin(object):
    def __init__(self, hub: str, owner: str, name: str, like: int | None = None, download: int | None = None):
        """

        :param hub: _description_
        :type hub: str
        :param owner: _description_
        :type owner: str
        :param name: _description_
        :type name: str

        :param like: _description_, defaults to None
        :type like: int | None, optional
        :param download: _description_, defaults to None
        :type download: int | None, optional

        Hub: The hub where the origin is located
        Owner: The owner of the origin
        Name: The name of the origin
        <Hub, Owner, Name> is the unique identifier of the origin

        Below are the properties of the Origin
        Like: The number of likes of the origin, it can be used to evaluate the popularity of the origin. Some origin do not have the like number.
        Download: The number of downloads of the origin, it can be used to evaluate the popularity of the origin. Some origin do not have the download number.

        """
        self._hub = hub
        self._owner = owner
        self._name = name

        self._like: int | None = None
        self._download: int | None = None

        self.setup_like(like)
        self.setup_download(download)

    def __hash__(self):
        return hash((self._hub, self._owner, self._name))

    def __str__(self):
        return f'Origin - <Hub/Owner/Name>: <{self._hub}/{self._owner}/{self._name}> (Like: {self._like}; Download: {self._download})'

    def __eq__(self, other: 'Origin') -> bool:
        return self._hub == other._hub and self._owner == other._owner and self._name == other._name

    def setup_like(self, like: int | None) -> None:
        self._like = like

    def setup_download(self, download: int | None) -> None:
        self._download = download

    @classmethod
    def load_dict(cls, d: dict) -> 'Origin':
        o = Origin(
            d['hub'],
            d['owner'],
            d['name'],
            like=d.get('like', None),
            download=d.get('download', None)
        )
        return o

    @classmethod
    def save_dict(cls, o: 'Origin') -> dict:
        d = dict(
            hub=o._hub,
            owner=o._owner,
            name=o._name,
            like=o._like,
            download=o._download
        )
        return d

    @classmethod
    def loads(cls, t: str) -> 'Origin':
        d = loads_json(t)
        o = cls.load_dict(d)
        return o

    @classmethod
    def saves(cls, o: 'Origin') -> str:
        d = cls.save_dict(o)
        t = saves_json(d)
        return t


class Evaluation(object):
    def __init__(self, task: str, dataset: str):
        """

        :param task: _description_
        :type task: str
        :param dataset: _description_
        :type dataset: str

        :param results: _description_, defaults to None
        :type results: dict[str, Any] | None, optional

        Origin: The origin of the evaluation
        Task: The task of the evaluation
        Dataset: The dataset of the evaluation
        <Origin, Task, Dataset> is the unique identifier of the evaluation

        Below are the properties of the Evaluation
        Results: The results of the evaluation, it is a dictionary of the metric name and the metric value.

        """
        self._task = task
        self._dataset = dataset

        self._results: dict[str, Any] = dict()
        self.clean_results()
        for metric_name, metric_value in results.items():
            self.insert_result(metric_name, metric_value)

    def __hash__(self):
        return hash((self._task, self._dataset))

    def __str__(self):
        return f'Evaluation - <Task/Dataset>: <{self._task}/{self._dataset}> (Results: {', '.join([f'{mn}: {mv}' for mn, mv in self._results.items()])})'

    def __eq__(self, other: 'Evaluation') -> bool:
        return self._task == other._task and self._dataset == other._dataset

    def clean_results(self) -> None:
        self._results = dict()

    def insert_result(self, metric_name: str, metric_value: Any) -> Any:
        self._results[metric_name] = metric_value
        return self._results[metric_name]

    def delete_result(self, metric_name: str) -> Any:
        metric_value = self._results.pop(metric_name)
        return metric_value

    def update_result(self, metric_name: str, metric_value: Any) -> None:
        self._results[metric_name] = metric_value
        return self._results[metric_name]

    @classmethod
    def load_dict(cls, d: dict) -> 'Evaluation':
        o = Origin(
            d['task'],
            d['dataset'],
            results=d.get('results', None)
        )
        return o

    @classmethod
    def save_dict(cls, e: 'Evaluation') -> dict:
        d = dict(
            task=e._task,
            dataset=e._dataset,
            results=e._results
        )
        return d

    @classmethod
    def loads(cls, t: str) -> 'Evaluation':
        d = loads_json(t)
        o = cls.load_dict(d)
        return o

    @classmethod
    def saves(cls, e: 'Evaluation') -> str:
        d = cls.save_dict(e)
        t = saves_json(d)
        return t


class Label(object):
    def __init__(self, origin: Origin, evaluation: Evaluation | None = None, results: dict[str, Any] | None = None):
        """

        :param records: _description_, defaults to None
        :type records: _type_, optional

        """
        self._records: dict[str, tuple[Origin, Evaluation]] = dict()
        self.clean_records()

    def clean_records(self) -> None:
        self._records = dict()

    def insert_record(self, origin: Origin, evaluation: Evaluation) -> None:
        if origin in self._records:
            pass
        else:
            self._records[origin] = set()

    def delete_origin(self, origin: Origin) -> None:
        if origin in self._records:
            self._records.pop(origin)
        else:
            pass

    def insert_evaluation(self, origin: Origin, evaluation: Evaluation) -> None:
        if origin in self._records:
            if evaluation in self._records[origin]:
                self._records[origin]
            else:
                self._records[origin].add(evaluation)
        else:
            self._records[origin] = set([evaluation])

    def delete_evaluation(self, origin: Origin, evaluation: Evaluation) -> None:
        if origin in self._records:
            if evaluation in self._records[origin]:
                self._records[origin].remove(evaluation)
            else:
                pass
        else:
            pass

    def insert_result(self, origin: Origin, evaluation: Evaluation, metric_name: str, metric_value: Any) -> None:
        if origin in self._records:
            if evaluation in self._records[origin]:
                self._records[origin][evaluation].insert_result(metric_name, metric_value)
            else:
                pass
        else:
            pass

    @classmethod
    def loads(cls, txt: str) -> 'Label':
        pass

    @classmethod
    def saves(cls, lbl: 'Label') -> str:
        pass
