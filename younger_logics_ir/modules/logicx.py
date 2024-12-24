#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-19 22:12:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-23 11:07:12
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib
import networkx

from typing import Literal, Generator

from younger.commons.io import load_pickle, save_pickle, loads_pickle, saves_pickle, loads_json, saves_json
from younger.commons.hash import hash_string


class LogicX(object):
    def __init__(self, src: Literal['onnx', 'core'] | None = None, dag: networkx.DiGraph | None = None):
        assert src in {'onnx', 'core'} or src is None, f'Argument \"src\" must be in {{"onnx", "core"}} instead \"{type(src)}\"!'
        assert isinstance(dag, networkx.DiGraph) or dag is None, f'Argument \"dag\" must be `networkx.DiGraph` instead \"{type(dag)}\"!'
        self._src = src
        self._dag = dag

        # If the LogicX is not standardized, the self._relationship is None.
        # If the LogicX is standardized:
        #     - If the LogicX is a sub-LogicX, the self._relationship is its parent's hash.
        #     - If the LogicX has no parent, the self._relationship is its own hash.
        self._relationship: str | None = None

    @property
    def valid(self) -> bool:
        return self._dag is not None and self._src is not None

    @property
    def src_valid(self) -> bool:
        return self._src is not None

    @property
    def dag_valid(self) -> bool:
        return self._dag is not None

    @property
    def src(self) -> Literal['onnx', 'core'] | None:
        return self._src

    @property
    def dag(self) -> networkx.DiGraph | None:
        return self._dag

    def setup_src(self, src: Literal['onnx', 'core']) -> None:
        assert src in {'onnx', 'core'}, f'Argument \"src\" must be in {{"onnx", "core"}} instead \"{type(src)}\"!'
        self._src = src

    def setup_dag(self, dag: networkx.DiGraph) -> None:
        assert isinstance(dag, networkx.DiGraph), f'Argument \"dag\" must be `networkx.DiGraph` instead \"{type(src)}\"!'
        self._dag = dag

    def load(self, logicx_filepath: pathlib.Path) -> None:
        assert logicx_filepath.is_file(), f'There is no \"LogicX\" can be loaded from the specified path \"{logicx_filepath.absolute()}\".'
        sdag = self.__class__.saves_dag(self._dag)
        ssrc = saves_pickle(self._src)
        save_pickle((sdag, ssrc), logicx_filepath)
        return 

    def save(self, logicx_filepath: pathlib.Path) -> None:
        assert not logicx_filepath.is_file(), f'\"LogicX\" can not be saved into the specified path \"{logicx_filepath.absolute()}\".'
        (ldag, lsrc) = load_pickle(logicx_filepath)
        self._src = loads_pickle(lsrc)
        self._dag = self.__class__.loads_dag(ldag)
        return

    def node_features(self, node_index: str) -> dict[str, str | dict]:
        """
        Get node features by node index.

        :param node_index: _description_
        :type node_index: str

        :return: _description_
        :rtype: dict[str, str | dict | int]

        Format - {node_type: str, node_attr: dict}
        """
        return self._dag.nodes[node_index]

    def node_type_feature(self, node_index: str) -> str:
        return self._dag.nodes[node_index]['node_type']

    def node_attr_feature(self, node_index: str) -> dict:
        return self._dag.nodes[node_index]['node_attr']

    def node_indices(self, node_type: Literal['input', 'output', 'operator', 'outer']) -> Generator[str, None, None]:
        assert self.valid, f'\"LogicX\" is invalid!'
        for node_index in self._dag.nodes:
            if self._dag.nodes[node_index]['node_type'] == node_type:
                yield node_index
            else:
                continue

    @property
    def input_indices(self) -> Generator[str, None, None]:
        return self.node_indices('input')

    @property
    def output_indices(self) -> Generator[str, None, None]:
        return self.node_indices('output')

    @property
    def operator_indices(self) -> Generator[str, None, None]:
        return self.node_indices('operator')

    @property
    def outer_indices(self) -> Generator[str, None, None]:
        return self.node_indices('outer')

    @property
    def standard(self) -> bool:
        return not (self._relationship is None)

    @property
    def relationship(self) -> str:
        assert self.standard, f'\"LogicX\" is not standardized!'
        return self._relationship

    @classmethod
    def loads_dag(cls, txt: str) -> networkx.DiGraph:
        data = loads_json(txt)
        dag = networkx.DiGraph()

        dag.graph = data.get("graph", dict())
        networkx.json_graph.node_link_graph
        for node_data in data['nodes']:
            node_index = node_data['node_index']
            node_features = node_data['node_features']
            dag.add_node(node_index, **node_features)
        for edge_data in data['edges']:
            tail_index = edge_data['tail_index']
            head_index = edge_data['head_index']
            edge_features = edge_data['edge_features']
            dag.add_edge(tail_index, head_index, **edge_features)

        return dag

    @classmethod
    def saves_dag(cls, dag: networkx.DiGraph) -> str:
        def get_object_with_sorted_containers(object):
            if isinstance(object, dict):
                return {key: get_object_with_sorted_containers(value) for key, value in sorted(object.items())}
            elif isinstance(object, list):
                return [get_object_with_sorted_containers(item) for item in object]
            else:
                return object

        data = dict(
            graph=dag.graph,
            nodes=[
                dict(
                    node_features=get_object_with_sorted_containers(node_features),
                    node_index=node_index,
                ) for node_index, node_features in sorted(dag.nodes(data=True), key=lambda x: x[0])
            ],
            edges=[
                dict(
                    edge_features=get_object_with_sorted_containers(edge_features),
                    tail_index=tail_index,
                    head_index=head_index,
                ) for tail_index, head_index, edge_features in sorted(dag.edges(data=True), key=lambda x: (x[0], x[1]))
            ]
        )

        txt = saves_json(data)
        return txt

    @classmethod
    def standardize(cls, logicx: 'LogicX') -> tuple['LogicX', list['LogicX']]:
        """
        .. todo::
            All Sub-Graphs Should Be Standardized To LogicX.
            Sub-Graphs Here is Attributes of an 'Operator' node.

        """

        logicx_sons: list['LogicX'] = list()
        logicx_descendants: list['LogicX'] = list()
        for operator_index in logicx.operator_indices:
            operator_attributes: dict[str, dict] = logicx.node_attr_feature(operator_index)['attributes']
            for oa_name, oa_attr in operator_attributes.items():
                if isinstance(oa_attr['value'], networkx.DiGraph):
                    logicx_son, logicx_son_descendants = cls.standardize(LogicX(logicx.src, oa_attr['value']))
                    logicx_sons.append(logicx_son)
                    logicx_descendants.extend(logicx_son_descendants)
                    operator_attributes[oa_name]['value'] = cls.hash(logicx_son)

                if isinstance(oa_attr['value'], list) and isinstance(oa_attr['value'][0], networkx.DiGraph):
                    assert all(isinstance(possible_subdag, networkx.DiGraph) for possible_subdag in oa_attr['value']), f'All subdags should be `networkx.DiGraph`!'
                    new_oa_attr_value: list[str] = list() # All change to hash
                    for subdag in oa_attr['value']:
                        logicx_son, logicx_son_descendants = cls.standardize(LogicX(logicx.src, subdag))
                        logicx_sons.append(logicx_son)
                        logicx_descendants.extend(logicx_son_descendants)
                        new_oa_attr_value.append(cls.hash(logicx_son))
                    operator_attributes[oa_name]['value'] = new_oa_attr_value

        logicx_hash = cls.hash(logicx)
        for logicx_son in logicx_sons:
            logicx_son._relationship = logicx_hash

        logicx._relationship = logicx_hash
        return logicx, logicx_sons + logicx_descendants

    @classmethod
    def hash(cls, logicx: 'LogicX') -> str:
        """
        Topology Hash (Hash)

        :param logicx: _description_
        :type logicx: LogicX

        :return: _description_
        :rtype: str
        """
        assert logicx.standard, f'\"LogicX\" is not standardized!'
        return networkx.weisfeiler_lehman_graph_hash(logicx.dag, edge_attr=None, node_attr='node_tuid', iterations=3, digest_size=16)

    @classmethod
    def uuid(cls, logicx: 'LogicX') -> str:
        """
        LogicX Object Hash (UUID)

        :param logicx: _description_
        :type logicx: LogicX

        :return: _description_
        :rtype: str
        """
        assert logicx.standard, f'\"LogicX\" is not standardized!'
        return hash_string(cls.saves_dag(logicx.dag))
