#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-08 08:58:05
# Author: Yikang Yang (杨怡康) & Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-08 10:17:26
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import ast
import pathlib
import networkx

from typing import Literal

from younger.commons.io import load_json
from younger_logics_ir.modules import LogicX


def graphviz_digraph(digraph: networkx.DiGraph, filepath: pathlib.Path, simplify: bool = False):
    import graphviz
    dot = graphviz.Digraph(comment='YLIR - DiGraph Visualization - Through Graphviz')
    dot.attr(rankdir='TB')

    simplified_node_indices = set()
    for node_index in digraph.nodes:
        mkrstr = f'ID: {node_index}\nTUID: {digraph.nodes[node_index]["node_tuid"]}'
        config = dict()
        if digraph.nodes[node_index]['node_type'] == 'input' or digraph.nodes[node_index]['node_type'] == 'output':
            config.update({
                'style': 'filled',
                'fillcolor': 'gray'
            })
        if digraph.nodes[node_index]['node_type'] == 'outer':
            config.update({
                'shape': 'egg',
                'style': 'filled',
                'fillcolor': 'darkred'
            })
        if digraph.nodes[node_index]['node_type'] == 'operator':
            op_type = ast.literal_eval(digraph.nodes[node_index]['node_tuid'][len('operator-'):])[0]
            if simplify and op_type == 'Constant':
                simplified_node_indices.add(node_index)
                continue
            else:
                config.update({
                    'shape': 'box',
                    'fillcolor': 'lightblue'
                })
        dot.node(node_index, mkrstr, **config)

    for tail_index, head_index in digraph.edges:
        if tail_index in simplified_node_indices or head_index in simplified_node_indices:
            continue
        else:
            dot.edge(tail_index, head_index)

    dot.render(filepath.with_suffix(''), cleanup=True, format="pdf")


def visualize_logicx(load_filepath: pathlib.Path, save_filepath: pathlib.Path, simplify: bool = False):
    logicx = LogicX()
    logicx.load(load_filepath)
    graphviz_digraph(logicx.dag, save_filepath, simplify=simplify)


def visualize_dag(load_filepath: pathlib.Path, save_filepath: pathlib.Path, simplify: bool = False):
    dic = load_json(load_filepath)
    dag = LogicX.loadd_dag(dic)
    graphviz_digraph(dag, save_filepath, simplify=simplify)


def main(mode: Literal['LogicX', 'DAG'], load_filepath: pathlib.Path, save_filepath: pathlib.Path, **kwargs) -> None:
    assert mode in {'LogicX', 'DAG'}

    if mode == 'LogicX':
        visualize_logicx(load_filepath, save_filepath, simplify=kwargs['simplify'])

    if mode == 'DAG':
        visualize_dag(load_filepath, save_filepath, simplify=kwargs['simplify'])
