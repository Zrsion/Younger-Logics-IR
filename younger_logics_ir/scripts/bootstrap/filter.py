#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-03-18 10:09:28
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


def check_instance(parameter: tuple[pathlib.Path, int]) -> pathlib.Path | None:
    path, opset_version = parameter
    instance = Instance()
    try:
        instance.load(path)

        if opset_version is not None and opset_version != get_opset_version(instance.logicx.dag.graph['opset_import']):
            return None
        else:
            return path
    except:
        return None


def standardize_instance(parameter: tuple[pathlib.Path, pathlib.Path]) -> tuple[Origin, int]:
    path, save_path = parameter
    instance = Instance()
    instance.load(path)
    instance, instance_sods = Instance.standardize(instance)

    instance.save(save_path.joinpath(f'{instance.unique}'))
    for instance_sod in instance_sods:
        instance_sod.save(save_path.joinpath(f'{instance_sod.unique}'))
    return (instance.labels[0].origin, len(instance_sods))


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path, opset_version: int | None = None, worker_number: int = 4):
    if opset_version:
        logger.info(f'Filter {opset_version} ONNX OPSET Version')
    else:
        logger.info(f'Filter All. ONNX OPSET Version Not Specified.')

    clean_parameters = list()
    for input_dirpath in input_dirpaths:
        logger.info(f'Scanning Instances Directory Path: {input_dirpath}')
        for instance_dirpath in input_dirpath.iterdir():
            clean_parameters.append((instance_dirpath, opset_version))

    logger.info(f'Total Instances To Be Filtered: {len(clean_parameters)}')
    standardize_paramenters = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(clean_parameters), desc='Filtering') as progress_bar:
            for index, path in enumerate(pool.imap_unordered(check_instance, clean_parameters), start=1):
                if path is not None:
                    standardize_paramenters.append((path, output_dirpath))
                progress_bar.update(1)
    logger.info(f'Total Instances Filtered: {len(standardize_paramenters)}')

    logger.info(f'Total Instances To Be Standardized: {len(standardize_paramenters)}')
    instance_count = 0
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(standardize_paramenters), desc='Standardizing') as progress_bar:
            for index, (origin, sod_count) in enumerate(pool.imap_unordered(standardize_instance, standardize_paramenters), start=1):
                instance_count += 1 + sod_count
                progress_bar.set_postfix({f'Current Model ID': f'{origin.hub}/{origin.owner}/{origin.name} - {sod_count}'})
                progress_bar.update(1)
    logger.info(f'Total Instances Standardized: {instance_count}')

    logger.info(f'Finished')
