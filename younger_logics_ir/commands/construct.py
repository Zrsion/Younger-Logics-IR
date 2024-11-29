#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 16:04:03
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-29 09:40:50
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

@click.group(name='construct')
def construct():
    pass


@construct.command()
@click.option('--dataset-dirpath', type=str, required=True)
@click.option('--save-dirpath', type=str, default='.')
@click.option('--max-inclusive-version', type=int, default=None)
def filter(dataset_dirpath, save_dirpath, max_inclusive_version):
    pass