#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-09 09:27:45
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-27 10:03:57
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_logics_ir.commands import equip_logger


@click.group(name='output')
def output():
    pass

@output.command(name='filter')
@click.option('--input-dirpaths',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), multiple=True, help='The directory where the data will be loaded.')
@click.option('--output-dirpath',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--opset-version',    required=False, type=int, default=None, help='Used to specify the instances that are equal to the specified version.')
@click.option('--worker-number',    required=False, type=int, default=1, help='The number of workers to use for processing.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_filter(
    input_dirpaths,
    output_dirpath,
    opset_version,
    worker_number,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.bootstrap import filter

    filter.main(input_dirpaths, output_dirpath, opset_version=opset_version, worker_number=worker_number)


@output.command(name='semantics')
@click.option('--input-names',      required=True,  type=str, multiple=True, help='The name of each dataset.')
@click.option('--input-filepaths',  required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), multiple=True, help='The filepath where the embeddings of each dataset will be loaded.')
@click.option('--output-dirpath',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--mode',             required=True,  type=click.Choice(['cos', 'mmd', 'cls'], case_sensitive=True), help='Indicates the type of data that needs to be semantically analyzed.')
@click.option('--standardize',      is_flag=True,   help='Indicates the type of data that needs to be semantically analyzed.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_statistics(
    input_names,
    input_filepaths,
    output_dirpath,
    mode,
    standardize,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.bootstrap import semantics

    semantics.main(input_names, input_filepaths, output_dirpath, mode, standardize)
