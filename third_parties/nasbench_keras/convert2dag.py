'''
This project is based on the "nasbench_keras" repository by lienching 
(https://github.com/lienching/nasbench_keras), with modifications for 
converting nasbench models to Younger instances.
'''

import os
import json
import onnx
import pathlib
import argparse
import multiprocessing
import tensorflow as tf

from tf2onnx import tf_loader, optimizer
from tf2onnx.tfonnx import process_tf_graph
from google.protobuf import text_format
from nasbench_keras import ModelSpec, build_keras_model, build_module

from younger_logics_ir.modules import Instance
from younger_logics_ir.modules import Origin
from younger_logics_ir.modules.label import Implementation
from younger_logics_ir.commons.constants import YLIROriginHub
from younger_logics_ir.converters import convert


def tf2onnx_main_export(model_path, output_path, opset, model_type = 'keras'):
    # [NOTE] The Code are modified based on the official tensorflow-onnx source codes. (https://github.com/onnx/tensorflow-onnx/blob/main/tf2onnx/convert.py [Method: main])
    assert model_type in {'saved_model', 'keras', 'tflite', 'tfjs', 'graph_def', 'from_checkpoint'}
    model_path = pathlib.Path(model_path)
    output_path = pathlib.Path(output_path)
    model_name = model_path.name
    tfjs_filepath = None
    tflite_filepath = None
    frozen_graph = None
    inputs = None
    outputs = None
    external_tensor_storage = None
    const_node_values = None
    
    if model_type == 'keras':
        frozen_graph, inputs, outputs = tf_loader.from_keras(
            model_path, inputs, outputs
        )

    with tf.device("/cpu:0"):
        with tf.Graph().as_default() as tf_graph:
            if model_type not in {'tflite', 'tfjs'}:
                tf.import_graph_def(frozen_graph, name='')
            graph = process_tf_graph(
                tf_graph,
                const_node_values=const_node_values,
                input_names=inputs,
                output_names=outputs,
                tflite_path=tflite_filepath,
                tfjs_path=tfjs_filepath,
                opset=opset
            )
            onnx_graph = optimizer.optimize_graph(graph, catch_errors=True)
            model_proto = onnx_graph.make_model(f'converted from {model_name}', external_tensor_storage=external_tensor_storage)
    save_protobuf(output_path, model_proto)


def save_protobuf(path, message, as_text=False):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


def write_log(log_path, message):
    with open(log_path, "a") as log_file: 
        log_file.write(message + "\n")
       
        
def get_converted_model_ids(log_path):
    converted_model_ids = []
    try:
        with open(log_path, "r") as log_file:
            for line in log_file:
                flag = line.split(":")[1].strip()
                if flag != "success":
                    continue
                model_id = line.split(":")[0].strip() 
                converted_model_ids.append(model_id)

    except FileNotFoundError:
        print(f"Log file not found, which will be created during code execution: {log_path}")
    return converted_model_ids


def convert_pipeline(params):
    # Adjacency matrix and nuberically-coded layer list
    model, model_id, onnx_dir, keras_dir, dag_dir, opset, config = params
    matrix, labels = model

    # Transfer numerically-coded operations to layers (check base_ops.py)
    labels = (['input'] + [config['available_ops'][l] for l in labels[1:-1]] + ['output'])
    try:
        # Module graph
        spec = ModelSpec(matrix, labels, data_format='channels_first')

        # Create module
        features = tf.keras.layers.Input((3,224,224), 1)
        net_outputs = build_keras_model(spec, features, labels, config)
        net = tf.keras.Model(inputs=features, outputs=net_outputs)

        # Save the module
        net.save(keras_dir.joinpath(f'{model_id}.keras'))

        # Convert the module to ONNX
        tf2onnx_main_export(keras_dir.joinpath(f'{model_id}.keras'), 
                            onnx_dir.joinpath(f'{model_id}.onnx'),
                            opset, 'keras')
        
        instance = Instance()
        onnx_model = onnx.load(onnx_dir.joinpath(f'{model_id}.onnx'))

        # Convert the onnx model to Younger instance
        instance.setup_logicx(convert(onnx_model))
        instance.insert_label(
            Implementation(
                origin = Origin(YLIROriginHub.NAS, 'NAS-Bench-101', model_id)
            )
        )
        instance.save(dag_dir.joinpath(instance.unique))
        
        return True, model_id

    except Exception as e:
        return False, model_id
        

def main(args):
    models_path = pathlib.Path(args.models_path)
    onnx_dir = pathlib.Path(args.onnx_dir)
    dag_dir = pathlib.Path(args.dag_dir)
    keras_dir = pathlib.Path(args.keras_dir)
    log_path = pathlib.Path(args.log_path)
    opset = args.opset
    limit = args.limit
    worker_numbers = args.worker_numbers

    config = {'available_ops' : ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                'stem_filter_size' : 128,
                'data_format' : 'channels_first',
                'num_stacks' : 3,
                'num_modules_per_stack' : 2,
                'num_labels' : 1000}

    with open(models_path, "rb") as f:
        models = json.load(f)
    converted_model_ids = get_converted_model_ids(log_path)

    params = list()
    for index, (key, value) in enumerate(models.items()):
        if str(key) in converted_model_ids:
            continue
        if limit != -1 and index == limit:
            break
        model = value
        model_id = str(key)
        params.append((model, model_id, onnx_dir, keras_dir, dag_dir, opset, config))

    with multiprocessing.Pool(worker_numbers) as pool:
        for index, (flag, model_id) in enumerate(pool.imap_unordered(convert_pipeline, params), start=1):
            if flag:
                write_log(log_path, f"{model_id}: success")
            else:
                write_log(log_path, f"{model_id}: failed")
            if args.clean_keras:
                keras_dir.joinpath(f'{model_id}.keras').unlink(missing_ok=True)
            if args.clean_onnx:
                onnx_dir.joinpath(f'{model_id}.onnx').unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NASBench-101 model to DAG.')
    parser.add_argument('--models-path', type=str, required=True, help='Path to the JSON file containing the graphs.')
    parser.add_argument('--onnx-dir', type=str, required=True, help='Directory to save the converted onnx model.')
    parser.add_argument('--dag-dir', type=str, required=True, help='Directory to save the converted dag model.')
    parser.add_argument('--keras-dir', type=str, required=True, help='Directory to save the keras model.')
    parser.add_argument('--log-path', type=str, required=True, help='Log the conversion details.')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version.')
    parser.add_argument('--limit', type=int, default=-1, 
                        help='To limit the number of models to convert. -1 means no limit.')
    parser.add_argument('--clean-keras', action='store_true', help='To clean the keras model after conversion.')
    parser.add_argument('--clean-onnx', action='store_true', help='To clean the onnx model after conversion.')
    parser.add_argument('--worker-numbers', type=int, default=1, help='Number of workers to convert the models.')
    args = parser.parse_args()
    main(args)

    
