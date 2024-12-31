#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-04-17 17:20:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-31 11:10:46
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib
import networkx

from typing import Literal

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger_logics_ir.modules import Dataset, Instance, LogicX


def statistics_graph(graph: networkx.DiGraph) -> dict[str, dict[str, int] | int]:
    graph_statistics = dict(
        num_operators = dict(),
        num_node = graph.number_of_nodes(),
        num_edge = graph.number_of_edges(),
    )

    for node_id in graph.nodes:
        node_features = graph.nodes[node_id]['features']
        node_identifier = Network.get_node_identifier_from_features(node_features)
        graph_statistics['num_operators'][node_identifier] = graph_statistics['num_operators'].get(node_identifier, 0) + 1
    return graph_statistics


def statistics_instances(load_dirpath: pathlib.Path, save_dirpath: pathlib.Path, plot: bool = False):
    """
    .. todo::
        In future, please implement this method.

    :param load_dirpath: _description_
    :type load_dirpath: pathlib.Path
    :param save_dirpath: _description_
    :type save_dirpath: pathlib.Path
    :param plot: _description_, defaults to False
    :type plot: bool, optional
    """
    pass


def statistics_logicxs(load_dirpath: pathlib.Path, save_dirpath: pathlib.Path, plot: bool = False):
    statistics = dict()
    logicxs = Dataset.drain_logicxs(load_dirpath)
    for logicx in logicxs:
        pass

    statistics_filepath = save_dirpath.joinpath(f'statistics_logicxs.json')
    save_json(statistics, statistics_filepath)
    logger.info(f'Statistics Saved into: {statistics_filepath}')


def main(load_dirpath: pathlib.Path, save_dirpath: pathlib.Path, granularity: Literal['Instance', 'LogicX'], plot: bool = False):
    if granularity == 'Instance':
        statistics_instances(load_dirpath, save_dirpath, plot=plot)

    if granularity == 'LogicX':
        statistics_logicxs(load_dirpath, save_dirpath, plot=plot)
