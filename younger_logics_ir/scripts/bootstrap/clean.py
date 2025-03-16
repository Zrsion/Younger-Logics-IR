#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-31 09:52:57
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger_logics_ir.modules import Dataset, LogicX


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path):
    hash2logicx: dict[str, LogicX] = dict() # Topology Hash -> LogicX
    hash2detail: dict[str, list[str]] = dict() # Topology Hash -> list[Instance]
    for input_dirpath in input_dirpaths:
        logger.info(f'Scanning Instances Directory Path: {input_dirpath}')
        instances = Dataset.drain_instances(input_dirpath)
        logger.info(f'Total Instances To Be Cleaned: {len(instances)}')

        logger.info(f'Cleaning Instances to LogicX ...')
        with tqdm.tqdm(total=len(instances), desc='Clean Intances') as progress_bar:
            for instance in instances:
                logicx_skeleton = LogicX.skeletonize(instance.logicx)
                logicx_skeleton_hash = LogicX.hash(logicx_skeleton)
                hash2logicx[logicx_skeleton_hash] = hash2logicx.get(logicx_skeleton_hash, logicx_skeleton)
                hash2detail[logicx_skeleton_hash] = hash2detail.get(logicx_skeleton_hash, list()).append(instance.unique)
                progress_bar.update(1)
    logger.info(f'Total Heterogeneous LogicX: {len(hash2logicx)}')

    details_filepath = output_dirpath.joinpath('details.json')
    logger.info(f'Saving Cleaned Details ...')
    save_json(hash2detail, details_filepath, indent=2)
    logger.info(f'Done')

    logicxs_dirpath = output_dirpath.joinpath('logicxs')
    logger.info(f'Saving Cleaned LogicXs ...')
    with tqdm.tqdm(total=len(hash2logicx), desc='Save LogicX') as progress_bar:
        for logicx_skeleton_hash, logicx_skeleton in hash2logicx.items():
            logicx_skeleton.save(logicxs_dirpath.joinpath(logicx_skeleton_hash))
            progress_bar.update(1)
    logger.info(f'Done')

    logger.info(f'Finished')
