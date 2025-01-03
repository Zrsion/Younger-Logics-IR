#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-03 15:44:26
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-03 15:50:45
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx
import click
import pathlib


from younger_logics_ir.modules import Instance, Implementation, Origin
from younger_logics_ir.converters import convert

from younger_logics_ir.commons.constants import YLIROriginHub


@click.command()
@click.option('--load-dirpath', type=pathlib.Path, required=True)
@click.option('--save-dirpath', type=pathlib.Path, required=True)
def main(load_dirpath: pathlib.Path, save_dirpath: pathlib.Path):

    for index, load_filepath in enumerate(load_dirpath.glob('*.onnx')):
        onnx_model = onnx.load(load_filepath)

        instance = Instance()

        # Convert the ONNX model to YLIR instance
        instance.setup_logicx(convert(onnx_model))
        instance.insert_label(
            Implementation(
                origin = Origin(YLIROriginHub.LOCAL, 'Test', f'Test_{index}')
            )
        )
        instance.save(save_dirpath.joinpath(instance.unique))


if __name__ == '__main__':
    main()