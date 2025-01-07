#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-07 17:20:28
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx
import networkx

from typing import Any, Literal, Callable
from functools import partial
from onnx.shape_inference import infer_shapes
from onnx.inliner import inline_local_functions


def get_operator_origin(op_type: str, domain: str) -> str:
    """
    Retrieving the source of the operator (whether it is from ONNX or another provider/organization).

    :param op_type: _description_
    :type op_type: str
    :param domain: _description_
    :type domain: str
    :return: _description_
    :rtype: str
    """

    try:
        onnx.defs.get_schema(op_type, domain=domain)
        origin = 'onnx'
    except:
        origin = domain or 'unknown'
    return origin


def get_onnx_opset_version() -> int:
    """
    Retrieving the current IR version.

    :return: _description_
    :rtype: int
    """

    return onnx.defs.onnx_opset_version()


def get_default_attributes_of_operator(op_type: str, max_inclusive_version: int, domain: str ='') -> dict[str, tuple[int, str]] | None:
    """
    Retrieving all attributes (with default values) of a specified operator 'op_type' within a given 'domain' for a specific IR version ('max_inclusive_version').
    The purpose of this method is to retrieve the OpSchema.

    :param op_type: _description_
    :type op_type: str
    :param max_inclusive_version: _description_
    :type max_inclusive_version: int
    :param domain: _description_, defaults to ''
    :type domain: str, optional

    :return: _description_
    :rtype: dict[str, tuple[int, str]] | None

    All attributes have default value only contain types - {<AttrType.FLOAT: 1>, <AttrType.INT: 2>, <AttrType.STRING: 3>, <AttrType.INTS: 7>, <AttrType.STRINGS: 8>}
    Thus, we only stringize/destringize all values with - (str(value)/ast.literal_eval(str(value)))
    """

    try:
        attributes = dict()
        schema = onnx.defs.get_schema(op_type, max_inclusive_version, domain=domain)
        for attribute_name, attribute_define in schema.attributes.items():
            attributes[attribute_name] = (attribute_define.type.value, str(onnx.helper.get_attribute_value(attribute_define.default_value)))
    except:
        attributes = None

    return attributes


def trans_string_string_entry_proto(string_string_entry_proto: onnx.StringStringEntryProto) -> dict:
    """
    Translating StringStringEntryProto into `dict` type which contains arguments that are used in `onnx.helper.set_metadata_props`.

    :param string_string_entry_proto: _description_
    :type string_string_entry_proto: onnx.StringStringEntryProto

    :return: _description_
    :rtype: dict

    """
    key: str = string_string_entry_proto.key
    value: str = string_string_entry_proto.value
    string_string_entry_proto_dict = dict(
        key = key,
        value = value
    )
    return string_string_entry_proto_dict


def trans_training_info(training_info: onnx.TrainingInfoProto) -> Any:
    """
    Translating TrainingProto into `dict` type which contains arguments that are used in `onnx.helper.make_training_info`.

    :param training_info: _description_
    :type training_info: onnx.TrainingInfoProto

    :return: _description_
    :rtype: Any

    .. todo::
        Implement this method please!
    """

    pass


def trans_operator_set_id_proto(operator_set_id_proto: onnx.OperatorSetIdProto) -> dict:
    domain: str = operator_set_id_proto.domain
    version: int = operator_set_id_proto.version
    operator_set_id_proto_dict = dict(
        domain = domain,
        version = version
    )
    return operator_set_id_proto_dict


def trans_type_proto(type_proto: onnx.TypeProto) -> dict:
    """
    Translating TypeProto into `dict` type which contains arguments that are used in `onnx.helper.make_[some]_type_proto`.

    :param type_proto: _description_
    :type type_proto: onnx.TypeProto

    :raises NotImplementedError: _description_

    :return: _description_
    :rtype: dict

    Collect arguments that are used in `onnx.helper.make_[some]_type_proto`.
    onnx.helper.make_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
    onnx.helper.make_sparse_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto

    TypeProto
    The standard ONNX data types.
    .. code:: protobuf
        message TypeProto {
            message Tensor {
                optional int32 elem_type = 1;
                optional TensorShapeProto shape = 2;
            }
            message Sequence { // repeated T
                optional TypeProto elem_type = 1;
            };
            message Map { // map<K,V>
                optional int32 key_type = 1; // This field MUST refer to an integral type ([U]INT{8|16|32|64}) or STRING
                optional TypeProto value_type = 2;
            };
            message Optional { // wrapper for Tensor, Sequence, or Map
                optional TypeProto elem_type = 1;
            };
            message SparseTensor {
                optional int32 elem_type = 1;
                optional TensorShapeProto shape = 2;
            }
        
            oneof value {
                Tensor tensor_type = 1;
                Sequence sequence_type = 4;
                Map map_type = 5;
                Optional optional_type = 9;
                SparseTensor sparse_tensor_type = 8;
            }
            optional string denotation = 6; // An optional denotation can be used to denote the whole type with a standard semantic description as to what is stored inside. Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition for pre-defined type denotations.
        }
    See more details at proto code where define the enum type 'TypeProto': https://github.com/postrational/onnx/blob/master/onnx/onnx.proto
    Map and Sequence of TypeProto are different with MapProto and SequenceProto
    For MapProto and SequenceProto see: https://github.com/onnx/onnx/blob/main/onnx/onnx-data.proto
    This file contains the proto definitions for MapProto and SequenceProto.
    These protos are used to represent the data structures of maps and sequence for use in test data or ModelProto.

    This project only support for SparseTensorType and TensorType now.
    TensorShapeProto
        message TensorShapeProto {
            message Dimension {
            oneof value {
                int64 dim_value = 1;
                string dim_param = 2;
            };
            };
            repeated Dimension dim = 1;
        }
    See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes

    .. todo::
        Add code for handling 'map_type,' 'opaque_type,' 'optional,' and 'sequence' to support classical ML operators.

    .. note::
        DNN-only implementations of ONNX MAY elect to not support non-tensor values as input and output to graphs and nodes.
        These types are needed to naturally support classical ML operators.
        DNN operators SHOULD restrict their input and output types to tensors.

    """

    denotation: str = type_proto.denotation

    type_names = {
        'tensor_type',
        'sparse_tensor_type',
        'sequence_type',
        'optional_type',
        'map_type',
    }

    for type_name in type_names:
        if type_proto.HasField(type_name):
            if type_name in {'tensor_type', 'sparse_tensor_type'}:
                type_field = getattr(type_proto, type_name)
                elem_type: int = type_field.elem_type
                shape: list[int | str] = list()
                for dim in type_field.shape.dim:
                    if dim.HasField('dim_param'):
                        shape.append(dim.dim_param)
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                type_proto_dict = dict(
                    elem_type = elem_type,
                    shape = shape,
                    shape_denotation = denotation,
                )
            elif type_name in {'sequence_type', 'optional_type'}:
                type_field = getattr(type_proto, type_name)
                elem_type: onnx.TypeProto = type_field.elem_type
                type_proto_dict = dict(
                    elem_type = trans_type_proto(elem_type),
                )
            elif type_name in {'map_type'}:
                type_field = getattr(type_proto, type_name)
                key_type: int = type_field.key_type
                value_type: onnx.TypeProto = type_field.value_type
                type_proto_dict = dict(
                    key_type = key_type,
                    value_type = trans_type_proto(value_type),
                )
            else:
                raise NotImplementedError(f'Support for handling other field ({type_name}) functionality is not yet available.')
            break
        else:
            type_proto_dict = dict()
    return type_proto_dict


def trans_value_info_proto(value_info_proto: onnx.ValueInfoProto) -> dict:
    """
    Translating ValueInfoProto into `dict` type which contains arguments that are used in `onnx.helper.make_value_info`.

    :param value_info_proto: _description_
    :type value_info_proto: onnx.ValueInfoProto

    :return: _description_
    :rtype: dict

    ValueInfoProto
    ValueInfoProto is mostly used in input, output, and value_info field of the onnx graph.
    Inputs and Outputs: Each main (top-level) graph MUST define the names, types and shapes of its inputs and outputs, which are specified as 'value_info' structures.
    The main graph inputs and outputs are required to have a shape, indicating the rank, even though the exact dimensions need not be specified.
    See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs

    .. code:: protobuf
        message ValueInfoProto { // Defines information on value, including the name, the type, and the shape of the value.
            optional string name = 1;     // namespace Value
            optional TypeProto type = 2;
            optional string doc_string = 3;
        }

        message TypeProto {
            Map: key_type, value_type
            Opaque: domain, name
            Optional: elem_type
            Sequence: elem_type
            SparseTensor: elem_type, shape // -> It is SparseTensorTypeProto, not SpareTensorProto
            Tensor: elem_type, shape  // -> It is TensorTypeProto, not TensorProto
        }

    There are 2 way to get ValueInfo with SparseTensorType or TensorType.
    1. Make ValueInfo from 'elem_type' and 'shape' directly.
      A. onnx.helper.make_tensor_value_info(name: str, elem_type: int, shape: Sequence[str | int | None] | None, doc_string: str = '', shape_denotation: List[str] | None = None) → ValueInfoProto.
      B. onnx.helper.make_sparse_tensor_value_info(name: str, elem_type: int, shape: Sequence[str | int | None] | None, doc_string: str = '', shape_denotation: List[str] | None = None) → ValueInfoProto.
    2. Make TypeProto from 'elem_type' and 'shape', then make ValueInfo from TypeProto.
      A. onnx.helper.make_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
      --> onnx.helper.make_value_info(name: str, type_proto: TypeProto, doc_string: str = '') → ValueInfoProto
      B. onnx.helper.make_sparse_tensor_type_proto(elem_type: int, shape: Sequence[str | int | None] | None, shape_denotation: List[str] | None = None) → TypeProto
      --> onnx.helper.make_value_info(name: str, type_proto: TypeProto, doc_string: str = '') → ValueInfoProto
    This project adopts the 2nd way because it allows the transformation of more types of TypeProto in the future.
    """

    name: str = value_info_proto.name
    doc_string: str = value_info_proto.doc_string
    type_proto = trans_type_proto(value_info_proto.type)
    value_info_proto_dict = dict(
        name = name,
        type_proto = type_proto,
        doc_string = doc_string,
    )
    return value_info_proto_dict


def trans_tensor_proto(tensor_proto: onnx.TensorProto, neglect_tensor_values: bool = True) -> dict:
    """
    Translating TensorProto into `dict` type which contains arguments that are used in `onnx.helper.make_tensor`.

    :param tensor_proto: _description_
    :type tensor_proto: onnx.TensorProto
    :param neglect_tensor_values: _description_, defaults to True
    :type neglect_tensor_values: bool, optional

    :raises NotImplementedError: _description_
    :raises NotImplementedError: _description_

    :return: _description_
    :rtype: dict

    TensorProto
    TensorProto is mostly used in initializer field of the onnx graph.
    Initializer can be the default value of an input or specifies a constant value.
      1. Defualt value of Input: When an initializer has the same name as a graph input, it specifies a default value for that input.
          When a name appears in both the initializer list and the graph input list, a runtime MAY allow a caller to specify a value for this (input) name overriding the value specified in the initializer and a runtime MAY allow users to omit specifying a value for this (input) name, choosing the value specified in the initializer.
      2. Constant Value: When an initializer has a name different from all graph inputs, it specifies a constant value.
          Names of constants that are not meant to be overridden by the caller should appear only in the initializer list and not in the graph input list.
      See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs & https://onnx.ai/onnx/repo-docs/IR.html#nodes

    Collect arguments that are used in onnx.helper.make_tensor(name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = False) → TensorProto.
      1. Two fields 'data_location' and 'external_data' are added to support for storing large tensor values.
         Where DataLocation is a new enum:
         enum DataLocation {
             MESSAGE = 0;
             RAW = 1;
             EXTERNAL = 2;
         }
         Later it is changed into
         enum DataLocation {
             DEFAULT = 0; // - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
             EXTERNAL = 1; // - EXTERNAL - data stored in an external location as described by external_data field.
         }
      2. 'name,' 'data_type,' 'raw_data,' and 'data_location' are 'presence' fields.
      3. Field 'dims' has defualt value - []
      See more details at ONNX Issues: https://github.com/onnx/onnx/issues/5608 and https://github.com/onnx/onnx/pull/678
      See more details at proto code where define the enum type 'DataLocation': https://github.com/postrational/onnx/blob/master/onnx/onnx.proto

      `vals` in dict can be any type of data defined in onnx.TensorProto.DataType enum.
      There is an easy alternative method:
        1. Make a TensorProto by using onnx.numpy_helper.from_array(arr: ndarray, name: str | None = None) → TensorProto
        2. Make a Numpy Arrray by using onnx.numpy_helper.to_array(tensor: TensorProto, base_dir: str = '') → ndarray

    .. todo::
        Add code for handling 'external_data'
    """

    name: str = tensor_proto.name
    data_type: int = tensor_proto.data_type
    dims: list[int] = list(tensor_proto.dims)

    if not neglect_tensor_values:
        if tensor_proto.data_location == onnx.TensorProto.DataLocation.DEFAULT:
            if tensor_proto.raw_data:
                raw: bool = True
                vals = tensor_proto.raw_data
            else:
                raw: bool = False
                field = onnx.helper.tensor_dtype_to_field(data_type)
                vals = list(getattr(tensor_proto, field))

        elif tensor_proto.data_location == onnx.TensorProto.DataLocation.EXTERNAL:
            raise NotImplementedError(f'Support for handling external_data functionality is not yet available.')
        else:
            raise NotImplementedError(f'The ONNX proto file may have been updated, and the project does not yet support processing this type of data_location: {tensor_proto.data_location}.')
    else:
        raw: bool = False
        vals = None

    tensor_proto_dict = dict(
        name = name,
        data_type = data_type,
        dims = dims,
        vals = vals,
        raw = raw,
    )
    return tensor_proto_dict


def trans_sparse_tensor_proto(sparse_tensor_proto: onnx.SparseTensorProto, neglect_tensor_values: bool = True) -> dict:
    """
    Translating SparseTensorProto into `dict` type which contains arguments that are used in `onnx.helper.make_sparse_tensor`.

    :param sparse_tensor_proto: _description_
    :type sparse_tensor_proto: onnx.SparseTensorProto
    :param neglect_tensor_values: _description_, defaults to True
    :type neglect_tensor_values: bool, optional

    :return: _description_
    :rtype: dict

    SparseTensorProto
    SparseTensorProto is mostly used in sparse_initializer field of the onnx graph.
    The sequence of non-default values are encoded as a tensor of shape [NNZ] (the Number of NonZero elements).
    The default-value is zero for numeric tensors, and empty-string for string tensors.
    .. code::
        message SparseTensorProto {
            optional TensorProto values = 1;
            optional TensorProto indices = 2;
            repeated int64 dims = 3;
        }
    See more details at proto code where define the 'SparseTensorProto': https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx.proto
    See more details at ONNX Official Doc: https://onnx.ai/onnx/api/classes.html#sparsetensorproto
    `values` in dict must have a non-empty name present which serves as a name for SparseTensorProto when used in sparse_initializer list.
    Collect arguments that are used in onnx.helper.make_sparse_tensor(values: TensorProto, indices: TensorProto, dims: Sequence[int]) → SparseTensorProto.
    """

    values = trans_tensor_proto(sparse_tensor_proto.values) if not neglect_tensor_values else None
    indices = trans_tensor_proto(sparse_tensor_proto.indices) if not neglect_tensor_values else None
    dims: list[int] = list(sparse_tensor_proto.dims)

    sparse_tensor_proto_dict = dict(
        name = values['name'],
        values = values,
        indices = indices,
        dims = dims
    )
    return sparse_tensor_proto_dict


def trans_attribute_proto(attribute_proto: onnx.AttributeProto, trans_graph_proto_method: Callable[[onnx.GraphProto, ], Any], neglect_tensor_values: bool = True) -> dict:
    """
    Translating AttributeProto into `dict` type which contains arguments that are used in `onnx.helper.make_attribute`.

    :param attribute_proto: _description_
    :type attribute_proto: onnx.AttributeProto
    :param trans_graph_proto_method: _description_
    :type trans_graph_proto_method: Callable[[onnx.GraphProto, ], Any]
    :param neglect_tensor_values: _description_, defaults to True
    :type neglect_tensor_values: bool, optional

    :return: _description_
    :rtype: dict

    Collect arguments that are used in onnx.helper.make_attribute(key: str, value: Any, doc_string: str | None = None, attr_type: int | None = None) → AttributeProto
    1. A named attribute containing either singular float, integer, string, graph, and tensor values, or repeated float, integer, string, graph, and tensor values.
    2. An AttributeProto MUST contain the name field, and *only one* of the following content fields, effectively enforcing a C/C++ union equivalent.
    3. Node attributes are used to pass literal (static) values to operators.
    4. An attribute MUST have only one of the value-carrying properties.

    .. code:: protobuf
        message AttributeProto {
            reserved 12, 16 to 19;
            reserved "v";
            enum AttributeType { // Note: this enum is structurally identical to the OpSchema::AttrType enum defined in schema.h.  If you rev one, you likely need to rev the other.
            UNDEFINED = 0;
            FLOAT = 1;
            INT = 2;
            STRING = 3;
            TENSOR = 4;
            GRAPH = 5;
            SPARSE_TENSOR = 11;
            TYPE_PROTO = 13;
        
            FLOATS = 6;
            INTS = 7;
            STRINGS = 8;
            TENSORS = 9;
            GRAPHS = 10;
            SPARSE_TENSORS = 12;
            TYPE_PROTOS = 14;
            }

            optional string name = 1;           // namespace Attribute, the name field MUST be present for this version of the IR.
            optional string ref_attr_name = 21; // This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
            optional string doc_string = 13; // A human-readable documentation for this attribute. Markdown is allowed.
        
            # 1. For 0.0.1 versions of the IR, this field was not defined, and implementations needed to use has_field heuristics to determine which value field was in use.
            # 2. For IR_VERSION 0.0.2 or later, this field MUST be set and match the f|i|s|t|... field in use.  This change was made to accommodate proto3 implementations.
            #    This Project only support IR version > 0.0.2
            optional AttributeType type = 20;   // discriminator that indicates which field below is in use

            # Exactly ONE of the following fields must be present for this version of the IR
            optional float f = 2;               // float
            optional int64 i = 3;               // int
            optional bytes s = 4;               // UTF-8 string
            optional TensorProto t = 5;         // tensor value
            optional GraphProto g = 6;          // graph
            optional SparseTensorProto sparse_tensor = 22;  // sparse tensor value
            optional TypeProto tp = 14;          // type proto
        
            repeated float floats = 7;          // list of floats
            repeated int64 ints = 8;            // list of ints
            repeated bytes strings = 9;         // list of UTF-8 strings
            repeated TensorProto tensors = 10;  // list of tensors
            repeated GraphProto graphs = 11;    // list of graph
            repeated SparseTensorProto sparse_tensors = 23; // list of sparse tensors
            repeated TypeProto type_protos = 15;// list of type protos
        }

    See more details at proto code where define the 'AttributeProto': https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#attributes
    """

    key: str = attribute_proto.name
    doc_string: str = attribute_proto.doc_string
    attr_type: int = attribute_proto.type

    single_types: set[int] = {
        onnx.defs.OpSchema.AttrType.FLOAT,
        onnx.defs.OpSchema.AttrType.INT,
        onnx.defs.OpSchema.AttrType.STRING,
        onnx.defs.OpSchema.AttrType.GRAPH,
        onnx.defs.OpSchema.AttrType.TENSOR,
        onnx.defs.OpSchema.AttrType.SPARSE_TENSOR,
        onnx.defs.OpSchema.AttrType.TYPE_PROTO,
    }

    repeat_types: set[int] = {
        onnx.defs.OpSchema.AttrType.FLOATS,
        onnx.defs.OpSchema.AttrType.INTS,
        onnx.defs.OpSchema.AttrType.STRINGS,
        onnx.defs.OpSchema.AttrType.GRAPHS,
        onnx.defs.OpSchema.AttrType.TENSORS,
        onnx.defs.OpSchema.AttrType.SPARSE_TENSORS,
        onnx.defs.OpSchema.AttrType.TYPE_PROTOS,
    }

    trans_method: dict[int, Callable[[Any]], Any] = {
        onnx.defs.OpSchema.AttrType.FLOAT: lambda x: x,
        onnx.defs.OpSchema.AttrType.FLOATS: lambda x: x,
        onnx.defs.OpSchema.AttrType.INT: lambda x: x,
        onnx.defs.OpSchema.AttrType.INTS: lambda x: x,
        onnx.defs.OpSchema.AttrType.STRING: lambda x: x,
        onnx.defs.OpSchema.AttrType.STRINGS: lambda x: x,

        onnx.defs.OpSchema.AttrType.GRAPH: trans_graph_proto_method,
        onnx.defs.OpSchema.AttrType.GRAPHS: trans_graph_proto_method,

        onnx.defs.OpSchema.AttrType.TENSOR: partial(trans_tensor_proto, neglect_tensor_values=neglect_tensor_values),
        onnx.defs.OpSchema.AttrType.TENSORS: partial(trans_tensor_proto, neglect_tensor_values=neglect_tensor_values),

        onnx.defs.OpSchema.AttrType.SPARSE_TENSOR: partial(trans_sparse_tensor_proto, neglect_tensor_values=neglect_tensor_values),
        onnx.defs.OpSchema.AttrType.SPARSE_TENSORS: partial(trans_sparse_tensor_proto, neglect_tensor_values=neglect_tensor_values),

        onnx.defs.OpSchema.AttrType.TYPE_PROTO: trans_type_proto,
        onnx.defs.OpSchema.AttrType.TYPE_PROTOS: trans_type_proto,
    }

    if attr_type in single_types:
        attribute_proto_value = onnx.helper.get_attribute_value(attribute_proto)
        value = trans_method[attr_type](attribute_proto_value)

    if attr_type in repeat_types:
        attribute_proto_values = onnx.helper.get_attribute_value(attribute_proto)
        value = list()
        for attribute_proto_value in attribute_proto_values:
            value.append(trans_method[attr_type](attribute_proto_value))

    attribute_proto_dict = dict(
        key = key,
        value = value,
        doc_string = doc_string,
        attr_type = attr_type
    )
    return attribute_proto_dict


def trans_node_proto(node_proto: onnx.NodeProto, opset_import: dict[str, int], trans_graph_proto_method: Callable[[onnx.GraphProto, ], networkx.DiGraph], neglect_tensor_values: bool = True) -> dict:
    """
    Translating NodeProto into `dict` type which contains arguments that are used in `onnx.helper.make_node`.

    :param node_proto: _description_
    :type node_proto: onnx.NodeProto
    :param opset_import: _description_
    :type opset_import: dict[str, int]
    :param trans_graph_proto_method: _description_
    :type trans_graph_proto_method: Callable[[onnx.GraphProto, ], Any]
    :param neglect_tensor_values: _description_, defaults to True
    :type neglect_tensor_values: bool, optional

    :raises exception: _description_

    :return: _description_
    :rtype: dict

    Collect arguments that are used in onnx.helper.make_node(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str | None = None, doc_string: str | None = None, domain: str | None = None, **kwargs: Any) → NodeProto
    .. code:: protobuf
        message NodeProto {
            repeated string input = 1;    // namespace Value
            repeated string output = 2;   // namespace Value
            
            optional string name = 3;     // namespace Node, this field MAY be absent in some version of the IR
            
            optional string op_type = 4;  // namespace Operator
            optional string domain = 7;   // namespace Domain
            
            repeated AttributeProto attribute = 5; // Named attributes, another form of operator parameterization, used for constant values rather than propagated values.
            
            optional string doc_string = 6;
        }
    See more details at proto code where define the 'NodeProto': https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#nodes

    The option of the schema inputs can be: 'Single,' 'Optional,' or 'Variadic.'
    'Optional':
      1. Omit the last input/output;
      2. Provide an empty name of the optional input/output;
      3. MUST provide names for the calculated optional outputs & MUST NOT provide names of those not calculated.
      See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/ONNXTypes.html
    'Variadic':
      1. The last input or output of an operator MAY be marked as variadic.
      2. If the last input of the schema is 'Variadic,' the length of inputs can be larger than the schema's.
      See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#variadic-inputs-and-outputs

    There are two ways to leave an optional input or output unspecified:
    1. The first, available only for trailing inputs and outputs, is to simply not provide that input;
    2. The second method is to use an empty string in place of an input or output name.

    Check Inputs
        For each variadic operator input, N or more node inputs must be specified where N is the minimum arity of the operator.
        Some operators have inputs that are marked as optional, which means that a referring node MAY forgo providing values for such inputs.

    Check Outputs
        For each variadic operator output, N or more node outputs must be specified where N is the minimum arity of the operator.
        Some operators have outputs that are optional. When an actual output parameter of an operator is not specified, the operator implementation MAY forgo computing values for such outputs.
        Each node referring to an operator with optional outputs MUST provide a name for each output that is computed and MUST NOT provide names for outputs that are not computed.

    .. note::
        1. A node input in a nested subgraph MAY refer to names introduced in outer graphs (as node outputs, graph inputs, or graph initializers).
        2. In the case of a nested subgraph, a node output name MUST be distinct from the names from the outer scopes that are visible in the nested subgraph.
        That means, in ONNX, nested subgraphs can interact with the outer graph through inputs (e.g., node outputs, graph inputs, or initializers), while ensuring that subgraph outputs are distinct from outer graph names to avoid interfering with the main graph's execution flow.

    .. todo::
        Code ignores the processing of all third-party operators, excluding those defined by the official ONNX specification and user-defined functions.
        In the future, functionality to handle these third-party operators will need to be added.

    """

    # Process inputs (each is a Hidden Edge Name)
    # Record Trap Index
    operands: dict[str, int] = dict()
    for node_input_index, node_input in enumerate(node_proto.input):
        if node_input == '':
            continue
        else:
            operands[node_input] = node_input_index

    # Process outputs (each is a Hidden Edge Name)
    # Record Emit Index
    results: dict[str, int] = dict()
    for node_output_index, node_output in enumerate(node_proto.output):
        if node_output == '':
            raise onnx.defs.SchemaError
        else:
            results[node_output] = node_output_index

    name: str = node_proto.name
    op_type: str = node_proto.op_type
    domain: str = node_proto.domain
    # attribute: expand by using trans_attribute_proto
    doc_string: str = node_proto.doc_string

    try:
        schema = onnx.defs.get_schema(op_type, max_inclusive_version=opset_import[domain], domain=domain)
    except onnx.defs.SchemaError:
        schema = None
    except Exception as exception:
        raise exception(f'Caught an exception: {exception}')

    if schema is None:
        since_version: int = 0
    else:
        since_version: int = schema.since_version
        if len(schema.inputs) > 0:
            last_input = schema.inputs[-1]
            if last_input.option.value == last_input.option.Optional.value:
                assert len(node_proto.input) <= len(schema.inputs)
            if last_input.option.value == last_input.option.Variadic.value:
                assert len(node_proto.input) >= len(schema.inputs) - 1
        if len(schema.outputs) > 0:
            last_output = schema.outputs[-1]
            if last_output.option.value == last_output.option.Optional.value:
                assert len(node_proto.output) <= len(schema.outputs)
            if last_output.option.value == last_output.option.Variadic.value:
                assert len(node_proto.output) >= len(schema.outputs) - 1

    # Process attributes
    # Record whether there is a subgraph
    node_proto_attributes: dict[str, dict] = dict()
    for node_proto_attribute in node_proto.attribute:
        node_proto_attribute = trans_attribute_proto(node_proto_attribute, trans_graph_proto_method=trans_graph_proto_method, neglect_tensor_values=neglect_tensor_values)
        key = node_proto_attribute.pop('key')
        node_proto_attributes[key] = node_proto_attribute

    node_proto_dict = dict(
        operands = operands,
        results = results,
        name = name,
        op_type = op_type,
        domain = domain,
        attributes = node_proto_attributes,
        doc_string = doc_string,
        since_version = since_version,
    )
    return node_proto_dict


def trans_graph_proto(ox_graph: onnx.GraphProto, depth: int | None = None, constant_names: set[str] | None = None, opset_import: dict[str, int] | None = None, neglect_tensor_values: bool = True, verbose: bool = False) -> networkx.DiGraph:
    """
    Recursively translate all `ox_graph`s into `nx_graph`s.

    :param ox_graph: _description_
    :type ox_graph: onnx.GraphProto
    :param opset_import: _description_, defaults to None
    :type opset_import: dict[str, int] | None, optional
    :param neglect_tensor_values: _description_, defaults to True
    :type neglect_tensor_values: bool, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional

    :raises KeyError: _description_

    :return nx_graph: _description_
    :rtype: networkx.DiGraph

    An onnx graph defines the computational logic of a model and is comprised of a parameterized list of nodes that form a directed acyclic graph based on their inputs and outputs.
    This is the equivalent of the "network" or "graph" in many deep learning frameworks.

    .. code:: protobuf
        message GraphProto {
            repeated NodeProto node = 1; // -> processed in this method
            optional string name = 2;
            repeated TensorProto initializer = 5; // -> see trans_tensor_proto_from_ox_to_nx
            repeated SparseTensorProto sparse_initializer = 15; // -> see trans_sparse_tensor_proto_from_ox_to_nx

            optional string doc_string = 10;
            repeated ValueInfoProto input = 11; // -> see trans_value_info_proto_from_ox_to_nx
            repeated ValueInfoProto output = 12; // -> see trans_value_info_proto_from_ox_to_nx
            
            repeated ValueInfoProto value_info = 13; // -> see trans_value_info_proto_from_ox_to_nx
            
            repeated TensorAnnotation quantization_annotation = 14;
            
            reserved 3, 4, 6 to 9;
            reserved "ir_version", "producer_version", "producer_tag", "domain";
        }
    'quantization_annotation' field carries information to indicate the mapping among a tensor and its quantization parameter tensors.
     For example:
       For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated, which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.

    First of all, the occurrence of a name as a node output is said to be a definition (also graph input and graph initializer);
    The occurrence of a name as a node input is said to be a use (also graph output);
      -> So, the name of a node output corresponds a unique node;
      -> So, the name of a node input corresponds multiple nodes;
      This project uniquely identifies each dataflow using the node index of endpoints and the index of its input of head and its output of tail.
    Within a namespace, each name MUST be unique for each given graph.
    Value namespace contains: node inputs & outputs, tensor values (if named), graph inputs, outputs.
    For the infomation about relationship between 'input,' 'output,' 'value_info,' and 'initializer', see ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs & https://onnx.ai/onnx/repo-docs/IR.html#names-within-a-graph

    Then, `graph_inputs,` `graph_outputs,` and `graph_dataflows` record all `ValueInfo` flowing in the graph, as explained below:
    Inputs And Outputs And Dataflows
      1. The output of a node can be sent as input to other nodes, and it may even serve as several different inputs for another node.
      2. This project call the value send from one node to another node as dataflow (which is an edge of the nx_graph);
      3. So, some dataflow that have the same tail endpoint have the same name of value_info;

    * all ox_graph.input are saved in nx_graph.graph['graph_inputs']
    * all ox_graph.output are saved in nx_graph.graph['graph_outputs']
    * all ox_graph.value_info are saved in nx_graph.graph['graph_dataflows']

    And, `graph_initializers` record all `TensorProto` or `SparseTensorProto` (which can be 'constants' or 'default value' of the graph), as explained below:
    Process Initializer And Sparse Initializer
    Initializer can be the default value of an input or specifies a constant value.
      1. Defualt value of Input: When an initializer has the same name as a graph input, it specifies a default value for that input.
          When a name appears in both the initializer list and the graph input list, a runtime MAY allow a caller to specify a value for this (input) name overriding the value specified in the initializer and a runtime MAY allow users to omit specifying a value for this (input) name, choosing the value specified in the initializer.
      2. Constant Value: When an initializer has a name different from all graph inputs, it specifies a constant value.
          Names of constants that are not meant to be overridden by the caller should appear only in the initializer list and not in the graph input list.
          For example, this can be used to initialize a weight or bias of an operator.
      See more details at ONNX Official Doc: https://onnx.ai/onnx/repo-docs/IR.html#graphs & https://onnx.ai/onnx/repo-docs/IR.html#nodes

    Thus, we store all `graph_inputs,` `graph_outputs,` `graph_dataflows,` and `graph_initializers` as the graph attributes.
    If one know the `name`, the corresponding `dict` can be find in these graph attributes.

    Value Infos And Node Outputs
    The number of Value Infos maybe less than the number of Node Outputs!!!

    Nodes
    All nodes of nx_graph contains 4 types, the type of node are specified by 'node_type,' the attr of node are specified by 'node_attr,' and the 'depth' of the graph that current node belongs to (start from 0, the outest graph's depth is 0):
      1. 'operator' nodes (>=1), which are recorded in 'node' field of ox_graph;
        -. node_type='operator'
        -. node_attr has the following keys (all values are extracted by using `trans_node_proto`):
          --. 'operands,'(Dict[node_input_name, node_input_index]) 'results,'(Dict[node_output_name, node_output_index]) 'name,' 'op_type,' 'domain,' 'attributes,' and 'doc_string.'
        -. depth=N
      2. 'input' node (>=1), which are recorded in 'input' field of ox_graph;
        -. node_type='input'
        -. node_attr only contains 1 attribute, 'input_name', which is a key in `graph_inputs.`
        -. depth=N
      3. 'output' node (>=1), which are recorded in 'output' field of ox_graph;
        -. node_type='output'
        -. node_attr only contains 1 attribute, 'output_name', which is a key in `graph_outputs.`
        -. depth=N
      4. 'outer' node (>=1), whose name only appear in subgraphs, are the node outputs, graph inputs, or graph initializers in graph (see below);
        -. node_type='outer'
        -. node_attr only contains 1 attribute, 'outer_name', which is a key in its outer graph's `graph_inputs,` `graph_initializers,` and `node_outputs.`
        -. depth=N

    Edges
    All edges of nx_graph contains 3 types, but there is no need to explicitly distinguish between them:
      An edge start from a node which label is 'tail_index' and end to a node which label is 'head_index';
        a. 'head_index' records the index of the node to which the data flow is input.
        b. 'tail_index' records the index of the node from which the data flow output.
      They all have the attribute 'connections': set[tuple['trap_index'(int), 'emit_index'(int)]]
        a. 'trap_index' records which occurrence of "operand" in the node specified by 'head_index' received the data. (If the head is graph_output, trap_index is int(0).)
        b. 'emit_index' records which occurrence of 'result' in the node specified by 'tail_index' emitted the data. (If the tail is graph_input, emit_index is int(0).)
      1. node to node, recorded in 'value_info' field of ox_graph;
      2. input to node, recorded in 'input' field of ox_graph;
      3. node to output, recorded in 'output' field of ox_graph;

    Fixed Values
    All fixed values are the union of `node outputs`, `graph_inputs`, `graph_initializers`. The word "Fix" is valid only in relation to a Graph, as opposed to a Subgraph.

    .. note::
        About Subgraph Inputs And Outputs
        1. A node input in a nested subgraph MAY refer to names introduced in graphs (such as node outputs, graph inputs, or graph initializers).
        2. In the case of a nested subgraph, a node output name MUST be distinct from the names from the outer scopes that are visible in the nested subgraph.

        That means, in ONNX, nested subgraphs can interact with the outer graph through inputs (e.g., node outputs, graph inputs, or initializers), while ensuring that subgraph outputs are distinct from outer graph names to avoid interfering with the main graph's execution flow.

    .. todo::
        Some `value_info` may not be inferred by the `onnx.shape_inference.infer_shapes` method.
        This is because there are operators outside the official ONNX domain in the graph.
        Support for this part will need to be added in the future.
    """

    nx_graph = networkx.DiGraph()

    depth = depth or 0

    nx_graph.graph.update(
        dict(
            name = ox_graph.name,
            depth = depth
        )
    )

    constant_names = constant_names or set()
    opset_import = opset_import or dict()

    def add_node(node_type: Literal['input', 'output', 'operator', 'outer'], node_name: str, node_attr: dict[str, Any] | None = None) -> str:
        """
        .. todo::
            'outer' nodes indicate the node outputs, graph inputs, or graph initializers in the graph, which can be duplicated if the same name appears in multiple nodes of the subgraph.
            In the future, the project will need to add functionality to handle this situation (should be pointed from tha same 'outer' node).
            *Each node index follow the naming format: '{graph_name}-{depth}-{node_type}-{number_of_this_type_node}'*
        """
        assert node_type in {'input', 'output', 'operator', 'outer'}
        assert isinstance(node_attr, dict) or node_attr is None
        node_attr = node_attr or dict()

        # Topology Unique Identifier (TUID)
        if node_type == 'operator':
            op_type = node_attr['op_type']
            domain = node_attr['domain']
            since_version = node_attr['since_version']
            opset_version = opset_import[domain]
            node_tuid = f'{node_type}-{str((op_type, domain, since_version, opset_version))}'
        else:
            node_tuid = f'{node_type}'

        number_of_this_type_node_key = f'number_of_{node_type}s'
        number_of_this_type_node_value = nx_graph.graph.get(number_of_this_type_node_key, 0)
        node_index = f'{ox_graph.name}-{depth}-{node_type}-{number_of_this_type_node_value}'
        nx_graph.graph[number_of_this_type_node_key] = number_of_this_type_node_value + 1
        nx_graph.add_node(node_index, node_tuid=node_tuid, node_type=node_type, node_name=node_name, node_attr=node_attr)
        return node_index

    def add_edge(tail_index: str, head_index: str, emit_index: int, trap_index: int) -> None:
        connection = (emit_index, trap_index)
        if nx_graph.has_edge(tail_index, head_index):
            nx_graph.edges[tail_index, head_index]['connections'].add(connection)
        else:
            nx_graph.add_edge(tail_index, head_index, connections=set([connection]))

    # Save Input, Output, and Dataflow Into Dict
    # Key: Name in Value namespace; Value: node inputs & outputs, tensor values (if named, tensor_type_proto), graph inputs, outputs. - All is ValueInfo
    graph_inputs: dict[str, dict] = dict()
    graph_outputs: dict[str, dict] = dict()
    graph_dataflows: dict[str, dict] = dict()

    for input_value_info in ox_graph.input:
        input_value_info = trans_value_info_proto(input_value_info)
        name = input_value_info.pop('name')
        graph_inputs[name] = input_value_info

    for output_value_info in ox_graph.output:
        output_value_info = trans_value_info_proto(output_value_info)
        name = output_value_info.pop('name')
        graph_outputs[name] = output_value_info

    for internode_value_info in ox_graph.value_info:
        internode_value_info = trans_value_info_proto(internode_value_info)
        name = internode_value_info.pop('name')
        graph_dataflows[name] = internode_value_info

    nx_graph.graph.update(
        dict(
            graph_inputs = graph_inputs,
            graph_outputs = graph_outputs,
            graph_dataflows = graph_dataflows,
        )
    )

    # Save Initializers Into Dict
    graph_initializers: dict[str, dict] = dict()
    for graph_initializer in ox_graph.initializer:
        graph_initializer = trans_tensor_proto(graph_initializer, neglect_tensor_values=neglect_tensor_values)
        name = graph_initializer.pop('name')
        graph_initializers[name] = graph_initializer

    for graph_sparse_initializer in ox_graph.sparse_initializer:
        graph_sparse_initializer = trans_sparse_tensor_proto(graph_sparse_initializer, neglect_tensor_values=neglect_tensor_values)
        name = graph_sparse_initializer.pop('name')
        graph_initializers[name] = graph_sparse_initializer

    nx_graph.graph.update(
        dict(
            graph_initializers = graph_initializers,
        )
    )

    constant_names = constant_names | (set(graph_initializers.keys()) - set(graph_inputs.keys()))

    # Record: Input Edge Name (ien) -> Node Index of Tail (nit)
    # For Input Edge and Output Edge, it has only one tail and only one head.
    ien2nit: dict[str, str] = dict()
    for node_name in graph_inputs.keys():
        node_index = add_node('input', node_name=node_name)
        ien2nit[node_name] = node_index

    # Record: Output Edge Name (oen) -> Node Index of Head (nih)
    # For Input Edge and Output Edge, it has only one tail and only one head.
    oen2nih: dict[str, str] = dict()
    for node_name in graph_outputs.keys():
        node_index = add_node('output', node_name=node_name)
        oen2nih[node_name] = node_index

    # For Hidden Edge, it has one tail and multiple head.
    # Hidden Head Info <head_index, trap_index> (hhi)
    # Record: Hidden Edge Name (hen) -> [ <Node Index of Head (nih), Trap Index of Head (tit)> ]
    # Hidden Tail Info <tail_index, emit_index> (hti)
    # Record: Hidden Edge Name (hen) -> <Node Index of Tail (nit), Emit Index of Tail (eit)>
    hen2hhis: dict[str, list[tuple[str, int]]] = dict()
    hen2hti: dict[str, tuple[str, int]] = dict()
    for node in ox_graph.node:
        node_attr = trans_node_proto(
            node,
            opset_import=opset_import,
            trans_graph_proto_method=partial(
                trans_graph_proto,
                depth=depth+1,
                constant_names=constant_names,
                opset_import=opset_import,
                neglect_tensor_values=neglect_tensor_values,
                verbose=verbose
            ),
            neglect_tensor_values=neglect_tensor_values
        )
        node_index = add_node('operator', node_name=node_attr['name'], node_attr=node_attr)
        for result_name, result_index in node_attr['results'].items():
            hen2hti[result_name] = (node_index, result_index)
        for operand_name, operand_index in node_attr['operands'].items():
            edges_to_add = edges_to_add_by_edge_name.get(operand_name, list())
            hen2hhis[operand_name] = hen2hhis.get(operand_name, list()).append((node_index, operand_index))

    # Add Edges
    # For Hidden Edge, it has one tail and multiple head.
    # For Input Edge and Output Edge, it has only one tail and only one head.
    for node_index, node_features in nx_graph.nodes.items():
        if node_features['node_type'] != 'operator':
            continue
        edges_tobe_added = list()
        for operand_name, hhis in hen2hhis.items():
            if operand_name in constant_names:
                continue
            if operand_name in ien2nit:
                edges_tobe_added.append((ien2nit[operand_name], 0, node_index, ))
                (tail_index, emit_index) = (ien2nit[operand_name], 0)
            else:
                if operand_name in hen2hti:
                    (tail_index, emit_index) = hen2hti[operand_name]
                else:
                    (tail_index, emit_index) = (add_node('outer', node_name=operand_name), 0)
            add_edge(tail_index, head_index, emit_index, trap_index)

        for result_name, (tail_index, emit_index) in hen2hti.items():
            if result_name in oen2nih:
                (head_index, trap_index) = (oen2nih[result_name], 0)
            else:
                if result_name in hen2hhi:
                    (head_index, trap_index) = hen2hhi[result_name]
                else:
                    raise KeyError(f'{result_name} not defined!')
            add_edge(tail_index, head_index, emit_index, trap_index)

    for (tail_index, head_index), edge_features in nx_graph.edges.items():
        nx_graph.edges[tail_index, head_index]['connections'] = sorted(list(edge_features['connections']), key=lambda x: (x[0], x[1]))

    return nx_graph


def trans_model_proto(model: onnx.ModelProto, neglect_tensor_values: bool = True, verbose: bool = False) -> networkx.DiGraph:
    """
    Tranlating onnx.ModelProto into networkx.DiGraph.

    :param model: _description_
    :type model: onnx.ModelProto
    :param neglect_tensor_values: _description_, defaults to True
    :type neglect_tensor_values: bool, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional

    :return nx_graph: _description_
    :rtype: networkx.DiGraph

    onnx.ModelProto (model) provides more metadata than onnx.GraphProto.
    The main purpose of the model structure is to associate metadata with a graph which contains all the executable elements.
    The metadata is used when first reading the model file, giving an implementation the information it needs in order to determine whether it will be able to execute the model, generate logging messages, error reports, etc.
    Further, the metadata is useful to tools, such as IDEs and model galleries, which need it for informing humans about a given model's purpose and characteristics.

    .. code:: protobuf
        :linenos:
        message ModelProto {
            optional int64 ir_version = 1; // The version of the IR this model targets. See Version enum above. This field MUST be present.
            repeated OperatorSetIdProto opset_import = 8; // All ModelProtos MUST have at least one entry that specifies which version of the ONNX OperatorSet is being imported.
            
            optional string producer_name = 2; // The name of the framework or tool used to generate this model. This field SHOULD be present to indicate which implementation/tool/framework emitted the model.
            optional string producer_version = 3; // The version of the framework or tool used to generate this model. This field SHOULD be present to indicate which implementation/tool/framework emitted the model.
            
            optional string domain = 4; // Together with `model_version` and GraphProto.name, this forms the unique identity of the graph. For example: `com.facebook.fair` or `com.microsoft.cognitiveservices`
            optional int64 model_version = 5;
            optional string doc_string = 6;
            optional GraphProto graph = 7;
            repeated StringStringEntryProto metadata_props = 14; // Named metadata values; keys should be distinct.
            repeated TrainingInfoProto training_info = 20;
            
            repeated FunctionProto functions = 25;
        };

    If IR version >= 3, the model must specify opset_import. If IR version < 3, the model cannot have any opset_import specified.
    Thus this method only support IR version >= 3.
    https://onnx.ai/onnx/api/checker.html#onnx.checker.check_model
    https://onnx.ai/onnx/repo-docs/Versioning.html#released-versions

    OperatorSetIdProto
    This is the type of attribute opset_import of class ModelProto.
    This attribute specifies the versions of operators used in the model.
    Every operator or node belongs to a domain. All operators for the same domain share the same version.
    https://onnx.ai/onnx/api/classes.html#operatorsetidproto
    It's schema can be loaded by using onnx.defs.get_schema(*args, **kwargs)
    1. get_schema(op_type: str, max_inclusive_version: int, domain: str = ‘’) -> onnx.onnx_cpp2py_export.defs.OpSchema
       Return the schema of the operator op_type and for a specific version.
    2. get_schema(op_type: str, domain: str = ‘’) -> onnx.onnx_cpp2py_export.defs.OpSchema
       Return the schema of the operator op_type and for a specific version.
    https://onnx.ai/onnx/api/defs.html#onnx.defs.get_schema

    .. todo::
        Parse training_info in future. Now we only record training_info.
        TrainingInfoProto stores information for training a model.
        In particular, this defines two functionalities: An initialization-step & A training-algorithm-step.
        1. Initialization resets the model back to its original state as if no training has been performed. Training algorithm improves the model based on input data. The semantics of the initialization-step is that the initializers in ModelProto.graph and in TrainingInfoProto.algorithm are first initialized as specified by the initializers in the graph, and then updated by the initialization_binding in every instance in ModelProto.training_info.
        2. The field algorithm defines a computation graph which represents a training algorithm’s step. After the execution of a TrainingInfoProto.algorithm, the initializers specified by update_binding may be immediately updated. If the targeted training algorithm contains consecutive update steps (such as block coordinate descent methods), the user needs to create a TrainingInfoProto for each step.
        https://onnx.ai/onnx/api/classes.html#traininginfoproto
    """

    assert isinstance(model, onnx.ModelProto), f'Argument \"model\" must be an ONNX Model Proto (onnx.ModelProto) instead \"{type(model)}\"!'
    assert 3 <= model.ir_version, f'IR Version {model.ir_version} Not Support! Only Accept 3 <= IR Version (1.0 <= ONNX Version).'

    model = inline_local_functions(model) # Expand all local functions of the model.
    model = infer_shapes(model) # Infer all shape of hiddens.

    ir_version: str = model.ir_version
    # opset_import: export and record by using trans_operator_set_id_proto
    producer_name: str = model.producer_name
    producer_version: str = model.producer_version
    domain: str = model.domain
    model_version: int = model.model_version
    doc_string: str = model.doc_string
    # graph: expand by using trans_graph_proto
    # metadata_props: export and record by using trans_string_string_entry_proto
    # training_info: export and record by using trans_training_info
    # functions: expand by using inline_local_functions

    opset_import: dict[str, int] = dict()
    for ox_model_opset_import in model.opset_import:
        ox_model_opset_import: dict[str, str | int] = trans_operator_set_id_proto(ox_model_opset_import)
        opset_import[ox_model_opset_import['domain']] = ox_model_opset_import['version']

    metadata_props: list[dict[str, str]] = list()
    for ox_model_metadata_props in model.metadata_props:
        ox_model_metadata_props: dict[str, str] = trans_string_string_entry_proto(ox_model_metadata_props)
        metadata_props.append(ox_model_metadata_props)

    training_info: list[Any] = list()
    for ox_model_training_info in model.training_info:
        ox_model_training_info: Any = trans_training_info(ox_model_training_info)
        training_info.append(ox_model_training_info)

    graph: networkx.DiGraph = trans_graph_proto(model.graph, opset_import=opset_import, neglect_tensor_values=neglect_tensor_values, verbose=verbose)

    graph_attributes = dict(
        ir_version = ir_version,
        opset_import = opset_import,
        producer_name = producer_name,
        producer_version = producer_version,
        domain = domain,
        model_version = model_version,
        doc_string = doc_string,
        metadata_props = metadata_props,
        training_info = training_info,
    )

    graph.graph.update(**graph_attributes)

    return graph
