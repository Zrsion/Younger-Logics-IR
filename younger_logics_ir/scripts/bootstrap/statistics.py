#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-04-17 17:20:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-03 16:59:19
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from typing import Any, Literal

from younger.commons.io import save_json, get_object_with_sorted_dict
from younger.commons.logging import logger

from younger_logics_ir.modules import Dataset, Instance, LogicX


def statistics_instances(input_dirpaths: pathlib.Path, output_dirpath: pathlib.Path):
    """
    .. todo::
        In future, please implement this method.

    :param load_dirpath: _description_
    :type load_dirpath: pathlib.Path
    :param save_dirpath: _description_
    :type save_dirpath: pathlib.Path
    """
    pass


def statistics_logicxs(input_dirpaths: pathlib.Path, output_dirpath: pathlib.Path):
    logicxs = (logicx for input_dirpath in input_dirpaths for logicx in Dataset.drain_logicxs(input_dirpath))

    ne_with_max_non = dict(
        number_of_nodes=0,
        number_of_edges=0
    ) # The (number of nodes, number of edge) with max number of nodes.
    ne_with_max_noe = dict(
        number_of_nodes=0,
        number_of_edges=0
    ) # The (number of nodes, number of edge) with max number of edges.

    operator_occurence: dict[str, int] = dict()
    detailed_statistics: dict[str, Any] = dict()
    for logicx in logicxs:
        logicx_hash = LogicX.hash(logicx)
        non = logicx.dag.number_of_nodes()
        noe = logicx.dag.number_of_edges()
        if (ne_with_max_non['number_of_nodes'] < non) or (ne_with_max_non['number_of_nodes'] == non and ne_with_max_non['number_of_edges'] < noe):
            ne_with_max_non = (non, noe)
        if (ne_with_max_noe['number_of_edges'] < noe) or (ne_with_max_noe['number_of_edges'] == noe and ne_with_max_noe['number_of_nodes'] < non):
            ne_with_max_noe = (non, noe)

        this_operator_occurence: dict[str, int] = dict()
        for operator_node_index in logicx.node_indices('operator'):
            tuid = logicx.node_tuid_feature(operator_node_index)
            operator_occurence[tuid] = operator_occurence.get(tuid, 0) + 1
            this_operator_occurence[tuid] = this_operator_occurence.get(tuid, 0) + 1

        this_statistics = dict(
            number_of_nodes = non,
            number_of_edges = noe,
            operator_occurence = get_object_with_sorted_dict(this_operator_occurence),
        )
        detailed_statistics[logicx_hash] = this_statistics

    statistics = dict(
        detail = detailed_statistics,
        overall = dict(
            ne_with_max_non = ne_with_max_non,
            ne_with_max_noe = ne_with_max_noe,
            operator_occurence = get_object_with_sorted_dict(operator_occurence),
        )
    )

    statistics_filepath = output_dirpath.joinpath(f'statistics_logicxs.json')
    save_json(statistics, statistics_filepath)
    logger.info(f'Statistics Saved into: {statistics_filepath}')


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path, granularity: Literal['Instance', 'LogicX']):
    if granularity == 'Instance':
        statistics_instances(input_dirpaths, output_dirpath)

    if granularity == 'LogicX':
        statistics_logicxs(input_dirpaths, output_dirpath)
