"""
To generate the model_infos for the DeepNets-1M dataset.
"""

import gc
import pathlib
import click
import torch
import tqdm
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.utils import set_seed

set_seed(1111) # To be consistent with config.py of ppuda 

@click.command()
@click.option('--save-dir', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the model_infos will be saved.')
@click.option('--data-dir', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data is stored.')
@click.option('--interval', required=False, type=int, default=50000, help='The interval at which the model_infos will be saved.')
@click.option('--split', required=False, type=str, default='train', help='The split of the data to be used. (train set equals to DeepNets-1M)')
def main(
    save_dir,
    data_dir,
    interval,
    split,
): 
    added_graphs = set() 
    graphs_queue = DeepNets1M.loader(split=split,
                                    nets_dir=data_dir,
                                    large_images=False,
                                    virtual_edges=50,
                                    arch=None)
    model_info_list = []
    save_id = 0
    print('len(graphs_queue): ', len(graphs_queue))    
    for index, graphs in enumerate(tqdm.tqdm(graphs_queue)):
        if index >= len(graphs_queue):
            break
        net_args, net_idx = graphs.net_args[0], graphs.net_inds[0]
        if index % interval == 0 and index > 0:
            model_info_list.sort(key=lambda x: x[0])
            torch.save(model_info_list, save_dir.joinpath(f'model_infos_{save_id}.pth'))
            model_info_list.clear() 
            save_id += 1
            gc.collect()

        model_info_list.append((net_idx, net_args))
        assert not net_idx in added_graphs
        added_graphs.add(net_idx)
    
    print('len(added_graps) and len(graphs_queue): ', len(added_graphs), ' and ', len(graphs_queue))
    assert len(added_graphs) == len(graphs_queue)

if __name__ == '__main__':
    main()