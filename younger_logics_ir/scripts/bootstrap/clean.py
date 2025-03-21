#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-03-21 15:09:08
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib
import networkx

from younger.commons.io import save_json, create_dir
from younger.commons.logging import logger

from younger_logics_ir.modules import Dataset, LogicX


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path):
    details_filepath = output_dirpath.joinpath('details.json')
    logicxs_dirpath = output_dirpath.joinpath('logicxs')
    create_dir(logicxs_dirpath)

    # Instances Must Be Standardized
    # Standardized Instance (St-I) [HAS] Standardized LogicX (St-L) ------Skeletonize------> Skeletonized LogicX (Sk-L)
    # Sk-L [1 -to- N] St-L
    #                 St-L [1 -to- N] St-I
    forest: networkx.DiGraph = networkx.DiGraph() # St-L Forest | Node is St-L's Hash
    rsti2rstl: dict[str, str] = dict() # Root St-I Hash -> Root St-L Hash
    skl2stls: dict[str, set[str]] = dict() # Sk-L Hash -> list[St-L Hash]
    stl2stis: dict[str, set[str]] = dict() # St-L Hash -> list[St-I Hash]

    for input_dirpath in input_dirpaths:
        logger.info(f'Scanning Instances Directory Path: {input_dirpath}')
        instances = Dataset.drain_instances(input_dirpath)
        # logger.info(f'Total Instances To Be Cleaned: {len(instances)}')

        logger.info(f'Cleaning Instances to LogicX ...')
        with tqdm.tqdm(desc='Clean Intances') as progress_bar:
            for instance in instances:
                std_lgx = instance.logicx
                std_lgx_hash = LogicX.hash(std_lgx)
                skt_lgx = LogicX.skeletonize(instance.logicx)
                skt_lgx_hash = LogicX.hash(skt_lgx)

                std_ins_hash = instance.unique
                if std_lgx.relationship != std_lgx_hash:
                    forest.add_edge(std_lgx.relationship, std_lgx_hash) # Father -> Son
                else:
                    rsti2rstl[std_ins_hash] = std_lgx_hash
                skl2stls.setdefault(skt_lgx_hash, set()).add(std_lgx_hash)
                stl2stis.setdefault(std_lgx_hash, set()).add(std_ins_hash)
                skt_lgx_savepath = logicxs_dirpath.joinpath(skt_lgx_hash)
                if not skt_lgx_savepath.is_file():
                    skt_lgx.save(skt_lgx_savepath)
                progress_bar.update(1)
    logger.info(f'Total Heterogeneous Skeleton LogicX: {len(skl2stls)}')
    logger.info(f'Total Heterogeneous Standard LogicX: {len(stl2stis)}')
    can_be_subgraph = list()
    for rsti, rstl in rsti2rstl.items():
        if rstl in forest.nodes() and forest.in_degree[rstl] != 0:
            can_be_subgraph.append(rsti)
    logger.info(f'LogicX Can Be Sub-: {len(can_be_subgraph)}')

    logger.info(f'Saving Cleaned Details ...')
    details = dict(
        forest = networkx.readwrite.json_graph.adjacency_data(forest),
        skl2stls = {skl: list(stls) for skl, stls in skl2stls.items()},
        stl2stis = {stl: list(stis) for stl, stis in stl2stis.items()},
        can_be_subgraph = can_be_subgraph
    )
    save_json(details, details_filepath, indent=2)
    logger.info(f'Done')

    logger.info(f'Finished')
