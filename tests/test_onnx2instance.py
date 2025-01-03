#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-03 15:44:26
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-03 23:12:19
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx
import click
import pathlib


from younger.commons.io import save_json
from younger.commons.logging import logger
from younger_logics_ir.modules import Instance, LogicX, Implementation, Origin
from younger_logics_ir.converters import convert

from younger_logics_ir.commons.json import YLIRJSONEncoder
from younger_logics_ir.commons.constants import YLIROriginHub


def find_bytes_in_dict(d):
    for key, value in d.items():
        if isinstance(value, bytes):
            print(f"Found bytes at key: {key}, value: {value}")
        elif isinstance(value, dict):
            find_bytes_in_dict(value)
        elif isinstance(value, list):
            find_bytes_in_list(value)

def find_bytes_in_list(lst):
    for i, item in enumerate(lst):
        if isinstance(item, bytes):
            print(f"Found bytes at index {i}: {item}")
        elif isinstance(item, dict):
            find_bytes_in_dict(item)
        elif isinstance(item, list):
            find_bytes_in_list(item)


@click.command()
@click.option('--load-dirpath', type=pathlib.Path, required=True)
@click.option('--save-dirpath', type=pathlib.Path, required=True)
def main(load_dirpath: pathlib.Path, save_dirpath: pathlib.Path):

    for index, load_filepath in enumerate(load_dirpath.glob('*.onnx')):
        onnx_model_filename = load_filepath.name.split(".")[0]
        logger.info(onnx_model_filename)
        onnx_model = onnx.load(load_filepath, load_external_data=False)

        instance = Instance()

        # Convert the ONNX model to YLIR instance
        logger.info('onnx loaded')
        dag = convert(onnx_model)
        logger.info(1111111)
        instance.setup_logicx(dag)
        logger.info(2222222)
        instance.insert_label(
            Implementation(
                origin = Origin(YLIROriginHub.LOCAL, 'Test', f'{index}_{onnx_model_filename}')
            )
        )
        logger.info(33333333)
        # sd = LogicX.saved_dag(instance.logicx.dag)

        # find_bytes_in_dict(sd)

        instance_unique = instance.unique
        logger.info(4444444444)
        instance.save(save_dirpath.joinpath(instance_unique))

        save_json(LogicX.saved_dag(instance.logicx.dag), save_dirpath.joinpath(instance_unique, f'{onnx_model_filename}.json'), indent=2, cls=YLIRJSONEncoder)
        logger.info(f'Saved - {instance_unique}')


if __name__ == '__main__':
    main()