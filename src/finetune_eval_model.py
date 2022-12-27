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

'''
Albert finetune and evaluation model script.
'''
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from src.albert_model import AlbertModel


class AlbertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 task_name=None):
        super(AlbertCLSModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.albert = AlbertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.task_name = task_name
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels

        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels,
                                weight_init=TruncatedNormal(config.initializer_range),
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.softmax = P.Softmax(axis=-1)
        self.argmax = P.Argmax(axis=-1)
        self.squeeze = P.Squeeze(axis=-1)

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = self.albert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        if self.training:
            cls = self.dropout(cls)
        logits = self.dense_1(cls)
        return logits


class AlbertSquadModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(AlbertSquadModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.albert = AlbertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_type)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.is_training = is_training
        # self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = (-1, config.hidden_size)
        self.origin_shape = (-1, config.seq_length, self.num_labels)
        self.transpose_shape = (-1, self.num_labels, config.seq_length)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.albert(input_ids, token_type_id, input_mask)
        sequence = self.reshape(sequence_output, self.shape)
        logits = self.dense1(sequence)
        logits = self.cast(logits, self.dtype)
        logits = self.reshape(logits, self.origin_shape)
        logits = self.transpose(logits, (0, 2, 1))
        logits = self.log_softmax(self.reshape(logits, (-1, self.transpose_shape[-1])))
        logits = self.transpose(self.reshape(logits, self.transpose_shape), (0, 2, 1))
        return logits


class AlbertRaceModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=4,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(AlbertRaceModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0

        self.albert = AlbertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.max_seq_length = config.seq_length
        self.compute_type = config.compute_type

        self.softmax = P.Softmax(axis=-1)
        self.is_training = is_training
        self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.get_shape = P.Shape()
        self.argmax = P.Argmax(axis=-1)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        # self.shape = (-1, config.seq_length)
        self.transpose_shape = (-1, self.num_labels, config.seq_length)
        self.dropout_race = nn.Dropout(1 - dropout_prob)
        self.race_dense = nn.Dense(self.hidden_size,
                                   1,
                                   weight_init=self.weight_init).to_float(self.compute_type)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        # bsz_per_core = self.get_shape(input_ids)[0]
        #
        # input_ids = self.reshape(input_ids, (bsz_per_core*self.num_labels,self.max_seq_length))
        # input_mask = self.reshape(input_mask, (bsz_per_core*self.num_labels,self.max_seq_length))
        # token_type_id = self.reshape(token_type_id, (bsz_per_core*self.num_labels,self.max_seq_length))

        _, pooled_output, _ = self.albert(input_ids, token_type_id, input_mask)
        if self.is_training:
            pooled_output = self.dropout_race(pooled_output)

        logits = self.race_dense(pooled_output)

        # logits = self.reshape(logits, (bsz_per_core,self.num_labels))

        probabilities = self.softmax(logits)
        predictions = self.cast(self.argmax(probabilities), mindspore.int32)
        return probabilities, logits, predictions


class AlbertSquadV2Model(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0,
                 use_one_hot_embeddings=False, start_n_top=5, end_n_top=5):
        super(AlbertSquadV2Model, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.albert = AlbertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense_st = nn.Dense(config.hidden_size, 1,
                                 weight_init=self.weight_init,
                                 has_bias=True).to_float(config.compute_type)
        self.dense_ed1 = nn.Dense(config.hidden_size * 2, config.hidden_size,
                                  weight_init=self.weight_init,
                                  has_bias=True,
                                  activation="tanh").to_float(config.compute_type)
        self.dense_ed2 = nn.Dense(config.hidden_size, 1,
                                  weight_init=self.weight_init,
                                  has_bias=True).to_float(config.compute_type)
        self.dense_ac1 = nn.Dense(config.hidden_size * 2, config.hidden_size,
                                  weight_init=self.weight_init,
                                  has_bias=True,
                                  activation="tanh").to_float(config.compute_type)
        self.dense_ac2 = nn.Dense(config.hidden_size, 1,
                                  weight_init=self.weight_init,
                                  has_bias=False).to_float(config.compute_type)

        self.max_seq_length = config.seq_length
        self.dtype = mstype.float32
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.softmax = P.Softmax(axis=-1)
        self.is_training = is_training
        # self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.squeeze = P.Squeeze(axis=-1)
        self.tile = P.Tile()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.top_k = P.TopK(sorted=True)
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.layer_norm = nn.LayerNorm((config.hidden_size,), begin_norm_axis=-1).to_float(self.dtype)
        self.concat = P.Concat(axis=-1)
        self.get_shape = P.Shape()
        self.zeros = ops.Zeros()
        self.shape = (-1, config.hidden_size)
        # self.origin_shape = (-1, config.seq_length, self.num_labels)
        # self.transpose_shape = (-1, self.num_labels, config.seq_length)

    def construct(self, input_ids, input_mask, token_type_id, start_positions, p_mask):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.albert(input_ids, token_type_id, input_mask)
        sequence_output = self.transpose(sequence_output, (1, 0, 2))
        start_logits = self.dense_st(sequence_output)
        start_logits = self.transpose(self.squeeze(start_logits), (1, 0))
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = self.log_softmax(start_logits_masked)
        return_dict = {}

        if self.is_training:
            start_positions = self.reshape(start_positions, (-1,))
            start_index = self.one_hot(start_positions,
                                       self.max_seq_length, self.on_value, self.off_value)
            start_features = Tensor(np.einsum("lbh,bl->bh", sequence_output.asnumpy(), start_index.asnumpy()))
            start_features = self.tile(start_features[None], (self.max_seq_length, 1, 1))
            end_logits = self.dense_ed1(self.concat((sequence_output, start_features)))
            end_logits = self.layer_norm(end_logits)
            end_logits = self.dense_ed2(end_logits)
            end_logits = self.transpose(self.squeeze(end_logits), (1, 0))
            end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
            end_log_probs = self.log_softmax(end_logits_masked)
            return_dict["start_log_probs"] = start_log_probs
            return_dict["end_log_probs"] = end_log_probs
        else:
            start_top_log_probs, start_top_index = self.top_k(
                start_log_probs, self.start_n_top)
            start_index = self.one_hot(start_top_index,
                                       self.max_seq_length, self.on_value, self.off_value)
            start_features = Tensor(np.einsum("lbh,bkl->bkh", sequence_output.asnumpy(), start_index.asnumpy()))
            end_input = self.tile(sequence_output[:, :, None], (1, 1, self.start_n_top, 1))
            start_features = self.tile(start_features[None],
                                       (self.max_seq_length, 1, 1, 1))
            end_input = self.concat((end_input, start_features))
            end_logits = self.dense_ed1(end_input)
            end_logits = self.layer_norm(end_logits)
            end_logits = self.dense_ed2(end_logits)
            end_logits = self.transpose(self.reshape(end_logits, (self.max_seq_length, -1, self.start_n_top)),
                                        (1, 2, 0))
            end_logits_masked = end_logits * (1 - p_mask[:, None]) - 1e30 * p_mask[:, None]

            end_log_probs = self.log_softmax(end_logits_masked)
            end_top_log_probs, end_top_index = self.top_k(end_log_probs, self.end_n_top)
            end_top_log_probs = self.reshape(end_top_log_probs, (-1, self.start_n_top * self.end_n_top))
            end_top_index = self.reshape(end_top_index, (-1, self.start_n_top * self.end_n_top))
            return_dict["start_top_log_probs"] = start_top_log_probs
            return_dict["start_top_index"] = start_top_index
            return_dict["end_top_log_probs"] = end_top_log_probs
            return_dict["end_top_index"] = end_top_index

        bsz = self.get_shape(input_ids)[0]
        cls_index = self.one_hot(self.zeros((bsz,), mindspore.int32),
                                 self.max_seq_length,
                                 self.on_value, self.off_value)
        cls_feature = Tensor(np.einsum("lbh,bl->bh", sequence_output.asnumpy(), cls_index.asnumpy()))

        # get the representation of START
        start_p = self.softmax(start_logits_masked)
        start_feature = Tensor(np.einsum("lbh,bl->bh", sequence_output.asnumpy(), start_p.asnumpy()))

        # note(zhiliny): no dependency on end_feature so that we can obtain
        # one single `cls_logits` for each sample
        ans_feature = self.concat((start_feature, cls_feature))
        ans_feature = self.dense_ac1(ans_feature)
        ans_feature = self.dropout(ans_feature)
        cls_logits = self.dense_ac2(ans_feature)
        cls_logits = self.squeeze(cls_logits)

        return_dict["cls_logits"] = cls_logits

        return return_dict
