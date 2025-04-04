#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-04 13:09:06
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib
import networkx
import multiprocessing

from younger.commons.io import create_dir, save_json
from younger.commons.logging import logger

from younger_logics_ir.modules import Instance, LogicX, Origin


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


def filter_instance(parameter: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]) -> tuple[Origin, int]:
    path, std_dirpath, skt_dirpath, pdg_dirpath = parameter
    instance = Instance()
    instance.load(path)
    instance_unique = instance.unique

    logicx, logicx_sods = LogicX.simplify(instance.logicx)
    org_logicxs = [logicx] + logicx_sods
    pedigree = networkx.DiGraph()

    for org_logicx in org_logicxs:
        std_logicx = LogicX.standardize(logicx)
        skt_logicx = LogicX.skeletonize(logicx)

        std_logicx_id = LogicX.hash(std_logicx)
        skt_logicx_id = LogicX.hash(skt_logicx)

        org_logicx_id = LogicX.luid(org_logicx)
        if org_logicx_id != org_logicx.relationship:
            pedigree.add_edge(org_logicx.relationship, org_logicx_id, standard=std_logicx_id, skeleton=skt_logicx_id)

        std_logicx_savepath = std_dirpath.joinpath(std_logicx_id)
        if not std_logicx_savepath.is_file():
            std_logicx.save(std_logicx_savepath)

        skt_logicx_savepath = skt_dirpath.joinpath(skt_logicx_id)
        if not skt_logicx_savepath.is_file():
            skt_logicx.save(skt_logicx_savepath)

    if len(pedigree) != 0:
        save_json(networkx.readwrite.json_graph.adjacency_data(pedigree), pdg_dirpath.joinpath(instance_unique), indent=2)
    return (instance.labels[0].origin, len(org_logicxs))


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path, opset_version: int | None = None, worker_number: int = 4):
    if opset_version:
        logger.info(f'Filter {opset_version} ONNX OPSET Version')
    else:
        logger.info(f'Filter All. ONNX OPSET Version Not Specified.')

    check_parameters = list()
    for input_dirpath in input_dirpaths:
        logger.info(f'Scanning Instances Directory Path: {input_dirpath}')
        for instance_dirpath in input_dirpath.iterdir():
            check_parameters.append((instance_dirpath, opset_version))

    std_dirpath = output_dirpath.joinpath('standard')
    skt_dirpath = output_dirpath.joinpath('skeleton')
    pdg_dirpath = output_dirpath.joinpath('pedigree')
    create_dir(std_dirpath)
    create_dir(skt_dirpath)
    create_dir(pdg_dirpath)

    logger.info(f'Total Instances To Be Filtered: {len(check_parameters)}')
    filter_paramenters = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(check_parameters), desc='Initial Filter - For Opset') as progress_bar:
            for index, path in enumerate(pool.imap_unordered(check_instance, check_parameters), start=1):
                if path is not None:
                    filter_paramenters.append((path, std_dirpath, skt_dirpath, pdg_dirpath))
                progress_bar.update(1)
    logger.info(f'Total Instances After Initial Opset Filter: {len(filter_paramenters)}')

    logger.info(f'Total Instances To Be Simplified - Standardize & Skeletonize: {len(filter_paramenters)}')
    instance_count = 0
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(filter_paramenters), desc='Simplify - Standardize & Skeleonize') as progress_bar:
            for index, (origin, sod_count) in enumerate(pool.imap_unordered(filter_instance, filter_paramenters), start=1):
                instance_count += 1 + sod_count
                progress_bar.set_postfix({f'Current Model ID': f'{origin.hub}/{origin.owner}/{origin.name} - {1 + sod_count}'})
                progress_bar.update(1)
    logger.info(f'Total Instances Standardized: {instance_count}')

    logger.info(f'Finished')
