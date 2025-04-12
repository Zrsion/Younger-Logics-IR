#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 16:04:03
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-12 20:03:26
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_logics_ir.commands import equip_logger


@click.group(name='assort')
def assort():
    pass


@assort.command(name='visualize')
@click.option('--mode',             required=True,  type=click.Choice(['LogicX', 'DAG'], case_sensitive=True), help='Indicates the type of data that needs to be visualized.')
@click.option('--load-filepath',    required=True,  type=click.Path(exists=True,  file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--save-filepath',    required=True,  type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--simplify',         is_flag=True,   help='Use to simplify graph.')
@click.option('--skeleton',         is_flag=True,   help='Use to skeleton graph.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def assort_visualize(
    mode,
    load_filepath,
    save_filepath,
    simplify,
    skeleton,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.tools import visualize

    kwargs = dict(
        simplify=simplify,
        skeleton=skeleton
    )

    visualize.main(mode, load_filepath, save_filepath, **kwargs)
