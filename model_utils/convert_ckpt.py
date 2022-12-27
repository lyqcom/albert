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
scripts for converting ckpt
"""
import json
from mindspore import Parameter, Tensor, save_checkpoint
from mindspore import dtype
from mindspore import ops

transpose = ops.Transpose()

param_dict = {}

with open('ckpt.json', 'r+') as file:
    content = file.read()
ckpt = json.loads(content)

print("***************")
# embedding
param_dict["albert.albert.albert_embedding_lookup.embedding_table"] = \
    Parameter(Tensor(ckpt['bert/embeddings/word_embeddings'], dtype.float32))

param_dict["albert.albert.albert_embedding_postprocessor.token_type_embedding.embedding_table"] = \
    Parameter(Tensor(ckpt['bert/embeddings/token_type_embeddings'], dtype.float32))

param_dict["albert.albert.albert_embedding_postprocessor.full_position_embedding.embedding_table"] = \
    Parameter(Tensor(ckpt['bert/embeddings/position_embeddings'], dtype.float32))

param_dict["albert.albert.albert_embedding_postprocessor.layernorm.gamma"] = \
    Parameter(Tensor(ckpt['bert/embeddings/LayerNorm/gamma'], dtype.float32))

param_dict["albert.albert.albert_embedding_postprocessor.layernorm.beta"] = \
    Parameter(Tensor(ckpt['bert/embeddings/LayerNorm/beta'], dtype.float32))

# encoder
param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.query_layer.weight"] = \
    Parameter(transpose(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel'], dtype.float32),
        (1, 0)))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.query_layer.bias"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.key_layer.weight"] = \
    Parameter(transpose(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel'], dtype.float32),
        (1, 0)))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.key_layer.bias"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.value_layer.weight"] = \
    Parameter(transpose(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel'], dtype.float32),
        (1, 0)))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.value_layer.bias"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.output.dense.weight"] = \
    Parameter(transpose(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel'], dtype.float32),
        (1, 0)))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.output.dense.bias"] = \
    Parameter(
        Tensor(ckpt["bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias"], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.output.layernorm.gamma"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.attention.output.layernorm.beta"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.intermediate.weight"] = \
    Parameter(transpose(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel'], dtype.float32),
        (1, 0)))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.intermediate.bias"] = \
    Parameter(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.output.dense.weight"] = \
    Parameter(transpose(
        Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel'],
               dtype.float32), (1, 0)))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.output.dense.bias"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias'],
                     dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.output.layernorm.gamma"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma'], dtype.float32))

param_dict["albert.albert.albert_encoder.group.0.inner_group.0.output.layernorm.beta"] = \
    Parameter(Tensor(ckpt['bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta'], dtype.float32))

param_dict["albert.albert.dense.weight"] = \
    Parameter(transpose(Tensor(ckpt['bert/pooler/dense/kernel'], dtype.float32), (1, 0)))

param_dict["albert.albert.dense.bias"] = \
    Parameter(Tensor(ckpt['bert/pooler/dense/bias'], dtype.float32))

param_dict["albert.albert.albert_encoder.dense_layer_2d.weight"] = \
    Parameter(transpose(Tensor(ckpt['bert/encoder/embedding_hidden_mapping_in/kernel'], dtype.float32), (1, 0)))

param_dict["albert.albert.albert_encoder.dense_layer_2d.bias"] = \
    Parameter(Tensor(ckpt['bert/encoder/embedding_hidden_mapping_in/bias'], dtype.float32))

param_dict["albert.cls1.output_bias"] = \
    Parameter(Tensor(ckpt['cls/predictions/output_bias'], dtype.float32))

param_dict["albert.cls1.layernorm.beta"] = \
    Parameter(Tensor(ckpt['cls/predictions/transform/LayerNorm/beta'], dtype.float32))

param_dict["albert.cls1.layernorm.gamma"] = \
    Parameter(Tensor(ckpt['cls/predictions/transform/LayerNorm/gamma'], dtype.float32))

param_dict["albert.cls1.dense.bias"] = \
    Parameter(Tensor(ckpt['cls/predictions/transform/dense/bias'], dtype.float32))

param_dict["albert.cls1.dense.weight"] = \
    Parameter(transpose(Tensor(ckpt['cls/predictions/transform/dense/kernel'], dtype.float32), (1, 0)))

param_dict["albert.cls2.dense.weight"] = \
    Parameter(Tensor(ckpt['cls/seq_relationship/output_weights'], dtype.float32))

param_dict["albert.cls2.dense.bias"] = \
    Parameter(Tensor(ckpt['cls/seq_relationship/output_bias'], dtype.float32))

param_dict["global_step"] = \
    Parameter(Tensor(ckpt['global_step'], dtype.float32))

# 把修改后的param_dict重新存储成checkpoint文件
save_list = []
# 遍历修改后的dict，把它转化成MindSpore支持的存储格式，存储成checkpoint文件
for key, value in param_dict.items():
    save_list.append({"name": key, "data": value.data})
save_checkpoint(save_list, "albert_base.ckpt")
