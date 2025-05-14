#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-04-09 21:42:35
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-16 11:05:45
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import numpy
import pandas
import hdbscan
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Literal
from sklearn.metrics.pairwise import cosine_distances, rbf_kernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from younger.commons.io import create_dir
from younger.commons.logging import logger


def extract_embedding_cos(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    centers = {k: v.mean(axis=0) for k, v in datasets.items()}

    names = list(datasets.keys())
    cos_matrix = pandas.DataFrame(index=names, columns=names)
    for a in names:
        for b in names:
            cos_matrix.loc[a, b] = cosine_distances(centers[a].reshape(1, -1), centers[b].reshape(1, -1))[0][0]
    cos_matrix = cos_matrix.astype(float)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cos_matrix, annot=True, cmap='Blues')
    plt.tight_layout()
    plt.savefig(output_dirpath.joinpath('embedding_cos_heatmap.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    # plt.title('Cosine Distance Between Dataset Embedding Centers')
    plt.savefig(output_dirpath.joinpath('embedding_cos_heatmap.png'))
    cos_matrix.to_csv(output_dirpath.joinpath('embedding_cos_matrix.csv'))


def compute_mmd(x: numpy.ndarray, y: numpy.ndarray, gamma=1.0, kernel: Literal['rbf'] = 'rbf'):
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def extract_embedding_mmd(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    names = list(datasets.keys())
    mmd_matrix = pandas.DataFrame(index=names, columns=names)
    for a in names:
        for b in names:
            mmd_matrix.loc[a, b] = compute_mmd(datasets[a], datasets[b])
    mmd_matrix = mmd_matrix.astype(float)

    plt.figure(figsize=(6, 5))
    sns.heatmap(mmd_matrix, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(output_dirpath.joinpath('embedding_mmd_heatmap.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    # plt.title('MMD Distance Between Dataset Embeddings')
    plt.savefig(output_dirpath.joinpath('embedding_mmd_heatmap.png'))
    mmd_matrix.to_csv(output_dirpath.joinpath('embedding_mmd_matrix.csv'))


def extract_embedding_mix_cls(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    sample_size = 1000 
    reducers = {
        'pca': PCA(n_components=2),
        'tsne': TSNE(n_components=2, init='pca', learning_rate='auto'),
        'umap': UMAP(n_components=2)
    }

    datasets = {name: data[numpy.random.choice(data.shape[0], min(sample_size, data.shape[0]), replace=False)] for name, data in datasets.items()}
    all_data = numpy.vstack(list(datasets.values()))
    
    dataset_names = []
    for name, data in datasets.items():
        dataset_names.extend([name] * len(data))
    dataset_names = numpy.array(dataset_names)

    unique_names = list(datasets.keys())
    marker_list = ['o', '^', 'P', 'D', 'X']
    markers = {name: marker_list[i % len(marker_list)] for i, name in enumerate(unique_names)}

    summary = []
    for method, reducer in reducers.items():
        reduced = reducer.fit_transform(all_data)
        reduced = MinMaxScaler().fit_transform(reduced)

        # ====== KMeans ======
        best_k = 2
        best_score = -1
        for k in range(2, min(20, len(reduced))):
            kmeans = KMeans(n_clusters=k, n_init='auto').fit(reduced)
            score = silhouette_score(reduced, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k

        final_kmeans = KMeans(n_clusters=best_k, n_init='auto').fit(reduced)
        df_kmeans = pandas.DataFrame(reduced, columns=['x', 'y'])
        df_kmeans['cluster'] = final_kmeans.labels_
        df_kmeans['dataset'] = dataset_names

        plt.figure(figsize=(13, 9))
        scatter = sns.scatterplot(
            x='x', y='y', hue='cluster', style='dataset', data=df_kmeans,
            palette='tab20', s=50, markers=markers
        )
        handles, labels = scatter.get_legend_handles_labels()

        style_labels = list(datasets.keys())
        style_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in style_labels]
        plt.legend().remove()
        if style_handles_labels:
            style_handles, style_labels = zip(*style_handles_labels)
            plt.legend(style_handles, style_labels, title="Dataset", loc='best', fontsize=16, title_fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dirpath.joinpath(f'mix_embedding_{method}_kmeans.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dirpath.joinpath(f'mix_embedding_{method}_kmeans.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()

        summary.append({
            'dataset': 'mix',
            'reducer': method,
            'method': 'kmeans',
            'n_clusters': best_k,
            'silhouette_score': best_score,
        })

        # ====== HDBSCAN ======
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        if n_clusters > 1:
            score = silhouette_score(reduced[cluster_labels != -1], cluster_labels[cluster_labels != -1])
        else:
            score = -1

        df_hdb = pandas.DataFrame(reduced, columns=['x', 'y'])
        df_hdb['cluster'] = cluster_labels
        df_hdb['dataset'] = dataset_names

        plt.figure(figsize=(13, 9))
        scatter = sns.scatterplot(
            x='x', y='y', hue='cluster', style='dataset', data=df_hdb,
            palette='tab20', s=50, markers=markers
        )
        handles, labels = scatter.get_legend_handles_labels()

        style_labels = list(datasets.keys())
        style_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in style_labels]
        plt.legend().remove()
        if style_handles_labels:
            style_handles, style_labels = zip(*style_handles_labels)
            plt.legend(style_handles, style_labels, title="Dataset", loc='best', fontsize=16, title_fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dirpath.joinpath(f'mix_embedding_{method}_hdbscan.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dirpath.joinpath(f'mix_embedding_{method}_hdbscan.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()

        summary.append({
            'dataset': 'mix',
            'reducer': method,
            'method': 'hdbscan',
            'n_clusters': n_clusters,
            'silhouette_score': score,
        })

    pandas.DataFrame(summary).to_csv(output_dirpath.joinpath('mix_embedding_cls_summary.csv'), index=False)



def extract_embedding_cls(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    reducers = {
        'pca': PCA(n_components=2),
        'tsne': TSNE(n_components=2, init='pca', learning_rate='auto'),
        'umap': UMAP(n_components=2)
    }

    summary = []
    for name, data in datasets.items():
        for method, reducer in reducers.items():
            reduced = reducer.fit_transform(data)
            reduced = MinMaxScaler().fit_transform(reduced)

            # ====== KMeans ======
            best_k = 2
            best_score = -1
            for k in range(2, min(11, len(reduced))):
                kmeans = KMeans(n_clusters=k, n_init='auto').fit(reduced)
                score = silhouette_score(reduced, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k

            final_kmeans = KMeans(n_clusters=best_k, n_init='auto').fit(reduced)
            df_kmeans = pandas.DataFrame(reduced, columns=['x', 'y'])
            df_kmeans['label'] = final_kmeans.labels_
            plt.figure(figsize=(5, 4))
            sns.scatterplot(x='x', y='y', hue='label', data=df_kmeans, palette='tab10', s=10)
            plt.legend().remove()
            plt.tight_layout()
            plt.savefig(output_dirpath.joinpath(f'{name}_embedding_{method}_kmeans.pdf'), format='pdf', dpi=300, bbox_inches='tight') 
            # plt.title(f'{name} - {method.upper()} + KMeans (k={best_k}, Silhouette={best_score:.2f})')
            filename_kmeans = output_dirpath.joinpath(f'{name}_embedding_{method}_kmeans.png')
            plt.savefig(filename_kmeans)
            plt.close()

            summary.append({
                'dataset': name,
                'reducer': method,
                'method': 'kmeans',
                'n_clusters': best_k,
                'silhouette_score': best_score,
            })

            # ====== HDBSCAN ======
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(reduced)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                score = silhouette_score(reduced[labels != -1], labels[labels != -1])
            else:
                score = -1

            df_hdb = pandas.DataFrame(reduced, columns=['x', 'y'])
            df_hdb['label'] = labels
            plt.figure(figsize=(5, 4))
            sns.scatterplot(x='x', y='y', hue='label', data=df_hdb, palette='tab10', s=10)
            plt.legend().remove()
            plt.tight_layout()
            plt.savefig(output_dirpath.joinpath(f'{name}_embedding_{method}_hdbscan.pdf'), format='pdf', dpi=300, bbox_inches='tight')  
            # plt.title(f'{name} - {method.upper()} + HDBSCAN (k={n_clusters}, Silhouette={score:.2f})')
            filename_hdb = output_dirpath.joinpath(f'{name}_embedding_{method}_hdbscan.png')
            plt.savefig(filename_hdb)
            plt.close()

            summary.append({
                'dataset': name,
                'reducer': method,
                'method': 'hdbscan',
                'n_clusters': n_clusters,
                'silhouette_score': score,
            })

    pandas.DataFrame(summary).to_csv(output_dirpath.joinpath('embedding_cls_summary.csv'), index=False)


def main(input_names: list[str], input_filepaths: list[pathlib.Path], output_dirpath: pathlib.Path, mode: Literal['cos', 'mmd', 'cls'], standardize: bool = False):
    datasets: dict[str, list[pathlib.Path]] = {input_name: pandas.read_csv(input_filepath).to_numpy() for input_name, input_filepath in zip(input_names, input_filepaths) }
    create_dir(output_dirpath)

    if standardize:
        scaler = StandardScaler()
        datasets = {k: scaler.fit_transform(v) for k, v in datasets.items()}

    if mode == 'cos':
        logger.info(f'... COS Statistics ...')
        extract_embedding_cos(datasets, output_dirpath)

    if mode == 'mmd':
        logger.info(f'... MMD Statistics ...')
        extract_embedding_mmd(datasets, output_dirpath)

    if mode == 'cls':
        logger.info(f'... CLS Statistics ...')
        # extract_embedding_cls(datasets, output_dirpath)
        extract_embedding_mix_cls(datasets, output_dirpath)
