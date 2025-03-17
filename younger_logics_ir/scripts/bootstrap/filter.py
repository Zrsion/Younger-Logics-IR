#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-03-17 16:54:00
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib
import multiprocessing

from younger.commons.logging import logger

from younger_logics_ir.modules import Dataset, Instance, Origin


def get_opset_version(opset_import: dict[str, int]) -> int | None:
    opset_version = opset_import.get('', None)
    return opset_version


def standardize_instance(parameter: tuple[str, int]) -> bool:
    path, opset_version = parameter
    instance = Instance()
    try:
        instance.load(path)

        if opset_version is not None and opset_version != get_opset_version(instance.logicx.dag.graph['opset_import']):
            valid = False
        else:
            valid = True
    except:
        valid = False

    return valid, path


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path, opset_version: int | None = None, worker_number: int = 4):
    if opset_version:
        logger.info(f'Filter {opset_version} ONNX OPSET Version')
    else:
        logger.info(f'Filter All. ONNX OPSET Version Not Specified.')

    parameters = list()
    for input_dirpath in input_dirpaths:
        logger.info(f'Scanning Instances Directory Path: {input_dirpath}')
        for instance_dirpath in input_dirpath.iterdir():
            parameters.append((instance_dirpath, opset_version))

    logger.info(f'Total Instances To Be Filtered: {len(parameters)}')

    instance = Instance()
    instances: list[Instance] = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Filtering') as progress_bar:
            for index, (valid, path) in enumerate(pool.imap_unordered(standardize_instance, parameters), start=1):
                instance.load(path)
                origin, (instance, instance_sods) = instance.labels[0].origin, Instance.standardize(instance)
                if valid:
                    instances.append(instance)
                    instances.extend(instance_sods)
                progress_bar.set_postfix({f'Current Model ID': f'{origin.hub}/{origin.owner}/{origin.name}'})
                progress_bar.update(1)
    logger.info(f'Total Instances Filtered: {len(instances)}')
    Dataset.flush_instances(instances, output_dirpath)
    logger.info(f'Finished')
