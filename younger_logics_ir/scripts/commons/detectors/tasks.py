#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-27 15:29:15
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import re


def detect_task(string: str) -> str:
    task = ''
    if re.search(r'\babstractive\b', string) and 'summarization' in string:
        task = 'abstractive-summarization'
    elif re.search(r'\baction\b', string) and 'classification' in string:
        task = 'action-classification'
    elif re.search(r'\banalogy questions\b', string):
        task = 'analogy-questions'
    elif re.search(r'\blanguage modeling\b', string):
        task = 'language-modeling'
    elif re.search(r'\bnamed entity recognition\b', string):
        task = 'named-entity-recognition'
    elif re.search(r'\bquestion answering\b', string):
        task = 'question-answering'
    elif re.search(r'\bsts\b', string) or re.search(r'\bsentence similarity\b', string):
        task = 'sentence-similarity'
    elif re.search(r'\bimageclassification\b', string) or re.search(r'\bimage classification\b', string):
        task = 'image-classification'
    # More Need To Be Write Here In Future.

    return task