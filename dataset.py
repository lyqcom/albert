# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in run_pretrain.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C


def create_albert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, batch_size=32):
    """create train dataset"""
    # apply repeat operations
    # 构成所有文件的列表
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if file_name.endswith(".mindrecord"):
            data_files.append(os.path.join(data_dir, file_name))
    # 批量读取数据文件
    data_set = ds.MindDataset(data_files,
                              columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                            "token_boundary", "masked_lm_positions", "masked_lm_ids",
                                            "masked_lm_weights"],
                              shuffle=ds.Shuffle.FILES if do_shuffle == "true" else False,
                              num_shards=device_num, shard_id=rank)

    # 获取epoch中的批数
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    # 创建数据操作对象，进行数据转换
    type_cast_int_op = C.TypeCast(mstype.int32)
    type_cast_float_op = C.TypeCast(mstype.float32)
    data_set = data_set.map(operations=type_cast_int_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_float_op, input_columns="masked_lm_weights")
    data_set = data_set.map(operations=type_cast_int_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_int_op, input_columns="next_sentence_labels")
    data_set = data_set.map(operations=type_cast_int_op, input_columns="token_boundary")
    data_set = data_set.map(operations=type_cast_int_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_int_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_int_op, input_columns="input_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    print("batch number of one epoch: {}".format(data_set.get_dataset_size()))
    print("repeat count: {}".format(data_set.get_repeat_count()))

    return data_set


def generator_classifier_train(data_features):
    for feature in data_features:
        yield (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_ids)


def create_classification_dataset(batch_size=1, repeat_count=-1,
                                  data_path=None, task_name=None, do_shuffle=True, is_training=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.MindDataset(data_path, shuffle=do_shuffle,
                              columns_list=["input_ids", "input_mask", "segment_ids",
                                            "label_id", "is_real_example"])

    type_cast_op_float = C.TypeCast(mstype.float32)
    # 获取epoch中的批数
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)

    data_set = data_set.map(operations=type_cast_op, input_columns="label_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op_float, input_columns="is_real_example")
    if is_training:
        data_set = data_set.repeat(repeat_count)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    print("batch number of one epoch: {}".format(data_set.get_dataset_size()))
    print("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


def generator_squad(data_features):
    for feature in data_features:
        yield (feature.input_ids, feature.input_mask, feature.segment_ids, feature.unique_id)


def generator_squad_train(data_features):
    for feature in data_features:
        yield (feature.input_ids, feature.input_mask, feature.segment_ids, feature.start_position,
               feature.end_position, feature.unique_id, feature.is_impossible)


def create_squad_dataset(batch_size=1, repeat_count=1, data_file_path=None, schema_file_path=None,
                         is_training=True, do_shuffle=True, is_v2=False):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)

    if is_training:
        columns_list = ["input_ids", "input_mask", "segment_ids", "start_position",
                        "end_position", "unique_id", "is_impossible"]
        if is_v2:
            columns_list.append("p_mask")
        data_set = ds.MindDataset(data_file_path,
                                  columns_list=columns_list,
                                  shuffle=do_shuffle)
        data_set = data_set.map(operations=type_cast_op, input_columns="start_position")
        data_set = data_set.map(operations=type_cast_op, input_columns="end_position")
        data_set = data_set.repeat(repeat_count)
    else:
        columns_list = ["input_ids", "input_mask", "segment_ids", "unique_id"]
        if is_v2:
            columns_list.append("p_mask")
        data_set = ds.MindDataset(data_file_path, shuffle=do_shuffle,
                                  columns_list=columns_list)
    # get the number of lines
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="unique_id")
    if is_v2:
        data_set = data_set.map(operations=type_cast_op, input_columns="p_mask")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    print("batch number of one epoch: {}".format(data_set.get_dataset_size()))
    print("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


def create_race_dataset(batch_size=1, repeat_count=1, data_file_path=None, schema_file_path=None,
                        do_shuffle=True):
    """create finetune or evaluation dataset"""
    data_set = ds.MindDataset(data_file_path,
                              columns_list=["input_ids", "input_mask", "segment_ids",
                                            "label_id", "is_real_example"],
                              shuffle=do_shuffle)
    type_cast_op = C.TypeCast(mstype.int32)
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="label_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="is_real_example")
    data_set = data_set.repeat(repeat_count)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    print("batch number of one epoch: {}".format(data_set.get_dataset_size()))
    print("repeat count: {}".format(data_set.get_repeat_count()))

    return data_set


if __name__ == '__main__':
    print(os.listdir("../data/pre_train"))
