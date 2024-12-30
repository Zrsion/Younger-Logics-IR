#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-30 21:35:32
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib
import multiprocessing

from younger.commons.logging import logger

from younger_logics_ir.modules import Dataset, Instance, LogicX, Origin


def standardize_instance(parameter: tuple[str, int]) -> tuple[Origin, Instance, list[Instance], bool]:
    path, opset_version = parameter
    instance = Instance()
    instance.load(path)

    if opset_version is not None and opset_version != instance.logicx.dag.graph['opset_import']:
        origin, (instance, instance_sods), valid = (instance.origin, (Instance(), list()), False)
    else:
        origin, (instance, instance_sods), valid = (instance.origin, Instance.standardize(instance), True)

    return origin, instance, instance_sods, valid


def main(load_dirpath: pathlib.Path, save_dirpath: pathlib.Path, worker_number: int = 4):
    logger.info(f'Scanning Instances Directory Path: {load_dirpath}')
    instances = Dataset.drain_instances(load_dirpath)
    logger.info(f'Total Instances To Be Filtered: {len(instances)}')

    logger.info(f'Cleaning Instances ...')
    heterogeneous_instances: dict[str, (Instance)] = dict()
    for instance in instances:
        assert len(instance.labels) == 1, f'Initial Instance Labels Must Be Single Instead {len(instance.labels)}!'
        instance_label = instance.labels[0]
        logicx_hash = LogicX.hash(instance.logicx)
        if logicx_hash not in heterogeneous_instances:
            heterogeneous_instances[logicx_hash] = instance
        else:
            heterogeneous_instances[logicx_hash].insert_label(instance_label)
    logger.info(f'Total Heterogeneous Instances: {len(heterogeneous_instances)}')

    instances: list[Instance] = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Filtering') as progress_bar:
            for index, (origin, instance, instance_sods, valid) in enumerate(pool.imap_unordered(standardize_instance, parameters), start=1):
                progress_bar.set_postfix({f'Current Model ID': f'{origin.hub}/{origin.owner}/{origin.name}'})
                progress_bar.update(1)
                if valid:
                    instances.append(instance)
                    instances.extend(instance_sods)
    logger.info(f'Total Instances Filtered: {len(instances)}')
    Dataset.flush_instances(instances, save_dirpath)
    logger.info(f'Finished')
