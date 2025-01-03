#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 15:57:08
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-03 15:40:10
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

from younger_logics_ir.commands.create import create
# from younger_logics_ir.commands.update import update
from younger_logics_ir.commands.output import output


@click.group(name='younger-logics-ir')
def main():
    pass


main.add_command(create, name='create')
# main.add_command(update, name='update')
main.add_command(output, name='output')


if __name__ == '__main__':
    main()
