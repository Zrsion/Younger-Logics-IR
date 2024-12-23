#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-23 09:13:48
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from younger.commons.io import load_json, save_json
from younger.commons.version import semantic_release, str_to_sem

from younger_logics_ir.modules.meta import Meta
from younger_logics_ir.modules.label import Label
from younger_logics_ir.modules.logicx import LogicX


class Instance(object):
    _meta_filename = 'meta'
    _struct_filename = 'struct.bin'
    _labels_filename = 'labels.lbl'
    _unique_filename = 'unique.pln'
    def __init__(self) -> None:
        self._meta: Meta = Meta(fresh_checker=self.fresh_checker)
        self._logicx: LogicX = LogicX()
        self._labels: list[Label] = list()
        self._unique = None

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def logicx(self) -> LogicX:
        return self._logicx

    @property
    def labels(self) -> set[Label]:
        return self._labels

    @property
    def unique(self) -> str:
        return self._unique

    def fresh_checker(self) -> bool:
        return self._unique == ''

    def load(self, instance_dirpath: pathlib.Path) -> None:
        assert instance_dirpath.is_dir(), f'There is no \"Instance\" can be loaded from the specified directory \"{instance_dirpath.absolute()}\".'
        meta_filepath = instance_dirpath.joinpath(self.__class__._meta_filename)
        self._load_meta(meta_filepath)
        logicx_filepath = instance_dirpath.joinpath(self.__class__._logicx_filename)
        self._load_logicx(logicx_filepath)
        labels_filepath = instance_dirpath.joinpath(self.__class__._labels_filename)
        self._load_labels(labels_filepath)
        unique_filepath = instance_dirpath.joinpath(self.__class__._unique_filename)
        self._load_unique(unique_filepath)
        return

    def save(self, instance_dirpath: pathlib.Path) -> None:
        assert not instance_dirpath.is_dir(), f'\"Instance\" can not be saved into the specified directory \"{instance_dirpath.absolute()}\".'
        meta_filepath = instance_dirpath.joinpath(self.__class__._meta_filename)
        self._save_meta(meta_filepath)
        logicx_filepath = instance_dirpath.joinpath(self.__class__._logicx_filename)
        self._save_logicx(logicx_filepath)
        labels_filepath = instance_dirpath.joinpath(self.__class__._labels_filename)
        self._save_labels(labels_filepath)
        unique_filepath = instance_dirpath.joinpath(self.__class__._unique_filename)
        self._save_unique(unique_filepath)
        return

    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        assert meta_filepath.is_file(), f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".'
        self._meta.load(meta_filepath)
        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        assert not meta_filepath.is_file(), f'\"Meta\" can not be saved into the specified path \"{meta_filepath.absolute()}\".'
        self._meta.save(meta_filepath)
        return

    def _load_logicx(self, logicx_filepath: pathlib.Path) -> None:
        assert logicx_filepath.is_file(), f'There is no \"LogicX\" can be loaded from the specified path \"{logicx_filepath.absolute()}\".'
        self._logicx.load(logicx_filepath)
        return

    def _save_logicx(self, logicx_filepath: pathlib.Path) -> None:
        assert not logicx_filepath.is_file(), f'\"LogicX\" can not be saved into the specified path \"{logicx_filepath.absolute()}\".'
        self._logicx.save(logicx_filepath)
        return

    def _load_labels(self, labels_filepath: pathlib.Path) -> None:
        assert labels_filepath.is_file(), f'There is no \"Lables\" can be loaded from the specified path \"{labels_filepath.absolute()}\".'
        self._labels = [Label.loads(s) for s in load_json(labels_filepath)]
        return

    def _save_labels(self, labels_filepath: pathlib.Path) -> None:
        assert not labels_filepath.is_file(), f'\"Labels\" can not be saved into the specified path \"{labels_filepath.absolute()}\".'
        save_json([Label.saves(l) for l in self._labels], labels_filepath)
        return

    def _load_unique(self, unique_filepath: pathlib.Path) -> None:
        assert unique_filepath.is_file(), f'There is no \"Unique ID\" can be loaded from the specified path \"{unique_filepath.absolute()}\".'
        self._unique = load_json(unique_filepath)
        return

    def _save_unique(self, unique_filepath: pathlib.Path) -> None:
        assert not unique_filepath.is_file(), f'\"Unique ID\" can not be saved into the specified path \"{unique_filepath.absolute()}\".'
        save_json(self._unique, unique_filepath)
        return

    def setup_logicx(self, logicx: LogicX) -> None:
        assert isinstance(logicx, LogicX), f'Argument \"logicx\" must be LogicX instead \"{type(logicx)}\"!'
        assert logicx.valid, f'Argument \"logicx\" must be valid!'
        if self.meta.is_fresh:
            self._logicx = logicx
            self._unique = LogicX.uuid(self._logicx)
        return

    def insert_label(self, label: Label) -> None:
        if self.meta.is_fresh:
            if label in self._labels:
                pass
            else:
                self._labels.append(label)
        return

    def delete_label(self, label: Label) -> None:
        if self.meta.is_fresh:
            if label in self._labels:
                self._labels.remove(label)
            else:
                pass
        return

    def insert(self) -> bool:
        if self.meta.is_fresh:
            return False
        else:
            self.meta.set_new()
            return True

    def delete(self) -> bool:
        if self.meta.is_fresh:
            return False
        else:
            self.meta.set_old()
            return True

    def release(self, version: semantic_release.Version) -> None:
        if self.meta.is_fresh or version == str_to_sem('0.0.0'):
            return

        if self.meta.release:
            if self.meta.is_old:
                self.meta.set_retired(version)
        else:
            if self.meta.is_new:
                self.meta.set_release(version)
        return
