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
Albert for finetune script.
'''
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
from src.albert_for_pre_training import clip_grad
from src.finetune_eval_model import AlbertCLSModel, AlbertSquadModel, AlbertRaceModel, AlbertSquadV2Model
from src.utils import CrossEntropyCalculation

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class AlbertFinetuneCell(nn.Cell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):

        super(AlbertFinetuneCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  is_real_example,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        # init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids,
                            is_real_example)

        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 is_real_example)

        self.optimizer(grads)
        return loss


class AlbertSquadCell(nn.Cell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(AlbertSquadCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        """BertSquad"""
        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        if not self.gpu_target:
            init = self.alloc_status()
            init = F.depend(init, loss)
            clear_status = self.clear_status(init)
            scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            init = F.depend(init, grads)
            get_status = self.get_status(init)
            init = F.depend(init, get_status)
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class AlbertSquadV2Cell(nn.Cell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(AlbertSquadV2Cell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  p_mask,
                  sens=None,):
        """BertSquad"""
        weights = self.weights
        # init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible,
                            p_mask)

        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 p_mask)

        self.optimizer(grads)
        return loss


class AlbertRaceCell(nn.Cell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(AlbertRaceCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self, input_ids,
                  input_mask,
                  segment_ids,
                  label_id,
                  is_real_example):
        """Race loss"""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            segment_ids,
                            label_id,
                            is_real_example)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 segment_ids,
                                                 label_id,
                                                 is_real_example)
        self.optimizer(grads)
        return loss


class AlbertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 task_name=None):
        super(AlbertCLS, self).__init__()
        self.albert = AlbertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.is_training = is_training
        self.task_name = task_name
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.onehot = ops.OneHot(axis=-1)
        self.sum = P.ReduceSum()
        self.mean = P.ReduceMean()
        self.dtype = config.dtype
        self.multiply = P.Mul()
        self.sub = P.Sub()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.square = P.Square()
        self.softmax = P.Softmax(axis=-1)
        self.argmax = P.Argmax(axis=-1)
        self.squeeze = P.Squeeze(axis=-1)

    def construct(self, input_ids, input_mask, token_type_id, label_id, is_real_example):
        """Classification loss"""
        logits = self.albert(input_ids, input_mask, token_type_id)
        if self.task_name != "sts-b":
            log_probs = self.log_softmax(logits)
            loss = self.loss(log_probs, label_id, self.num_labels)
        else:
            logits = self.squeeze(logits)
            per_example_loss = self.square(self.sub(logits, label_id))
            loss = self.mean(per_example_loss)
        return loss


class AlbertSquad(nn.Cell):
    '''
    Train interface for SQuAD finetuning task.
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(AlbertSquad, self).__init__()
        self.albert = AlbertSquadModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.seq_length = config.seq_length
        self.is_training = is_training
        # self.total_num = Parameter(Tensor([0], mstype.float32))
        # self.start_num = Parameter(Tensor([0], mstype.float32))
        # self.end_num = Parameter(Tensor([0], mstype.float32))
        self.sum = P.ReduceSum()
        self.equal = P.Equal()
        self.argmax = P.ArgMaxWithValue(axis=1)
        self.squeeze = P.Squeeze(axis=-1)

    def construct(self, input_ids, input_mask, token_type_id, start_position, end_position, unique_id, is_impossible):
        """interface for SQuAD finetuning task"""
        logits = self.albert(input_ids, input_mask, token_type_id)
        if self.is_training:
            unstacked_logits_0 = self.squeeze(logits[:, :, 0:1])
            unstacked_logits_1 = self.squeeze(logits[:, :, 1:2])
            start_loss = self.loss(unstacked_logits_0, start_position, self.seq_length)
            end_loss = self.loss(unstacked_logits_1, end_position, self.seq_length)
            total_loss = (start_loss + end_loss) / 2.0
        else:
            start_logits = self.squeeze(logits[:, :, 0:1])
            start_logits = start_logits + 100 * input_mask
            end_logits = self.squeeze(logits[:, :, 1:2])
            end_logits = end_logits + 100 * input_mask
            total_loss = (unique_id, start_logits, end_logits)
        return total_loss


class AlbertSquadV2(nn.Cell):
    '''
    Train interface for SQuAD finetuning task.
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0,
                 use_one_hot_embeddings=False, start_n_top=5, end_n_top=5):
        super(AlbertSquadV2, self).__init__()
        self.albert = AlbertSquadV2Model(config=config, is_training=is_training, num_labels=num_labels,
                                         dropout_prob=dropout_prob, use_one_hot_embeddings=use_one_hot_embeddings,
                                         start_n_top=start_n_top, end_n_top=end_n_top)
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.num_labels = num_labels
        self.get_shape = P.Shape()
        self.seq_length = config.seq_length
        self.is_training = is_training
        # self.total_num = Parameter(Tensor([0], mstype.float32))
        # self.start_num = Parameter(Tensor([0], mstype.float32))
        # self.end_num = Parameter(Tensor([0], mstype.float32))
        self.sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.equal = P.Equal()
        self.add = P.Add()
        self.argmax = P.ArgMaxWithValue(axis=1)
        self.squeeze = P.Squeeze(axis=-1)
        self.one_hot = P.OneHot()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.neg = P.Neg()
        self.CEloss = CrossEntropyCalculation()
        self.sigmoid = P.Sigmoid()
        self.dtype = mstype.float32
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)

    def construct(self, input_ids, input_mask, token_type_id, start_position, end_position, unique_id, is_impossible,
                  p_mask):
        """interface for SQuAD finetuning task"""
        logits = self.albert(input_ids, input_mask, token_type_id, start_position, p_mask)
        if self.is_training:
            start_log_probs = logits["start_log_probs"]
            start_loss = self.CEloss(start_log_probs, start_position, self.seq_length)
            end_log_probs = logits["end_log_probs"]
            end_loss = self.CEloss(end_log_probs, end_position, self.seq_length)
            total_loss = self.add(start_loss, end_loss) / 2.0
            cls_logits = logits["cls_logits"]
            is_impossible = self.reshape(is_impossible, (-1,))
            regression_loss = self.loss(self.cast(cls_logits, mindspore.float32),
                                        self.cast(is_impossible, mindspore.float32))

            regression_loss = self.reduce_mean(regression_loss)
            total_loss += regression_loss * 0.5
        else:
            start_top_index = logits["start_top_index"]
            start_top_log_probs = logits["start_top_log_probs"]
            end_top_index = logits["end_top_index"]
            end_top_log_probs = logits["end_top_log_probs"]
            cls_logits = logits["cls_logits"]
            total_loss = (unique_id, start_top_log_probs, start_top_index,
                          end_top_log_probs, end_top_index, cls_logits)
        return total_loss


class AlbertRace(nn.Cell):
    '''
    Train interface for SQuAD finetuning task.
    '''

    def __init__(self, config, is_training, num_labels=4,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(AlbertRace, self).__init__()
        self.albert = AlbertRaceModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.num_labels = num_labels
        self.loss = CrossEntropyCalculation(is_training)
        self.is_training = is_training
        self.sum = P.ReduceSum()
        self.max_seq_length = config.seq_length
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.mean = P.ReduceMean()
        self.cast = P.Cast()
        self.dtype = config.dtype
        self.reshape = P.Reshape()
        self.onehot = P.OneHot()
        self.multiply = P.Mul()
        self.neg = P.Neg()
        self.get_shape = P.Shape()
        self.reshape = P.Reshape()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)

    def construct(self, input_ids, input_mask, segment_ids, label_ids, is_real_example):
        """calculate loss"""
        bsz_per_core = self.get_shape(input_ids)[0]

        input_ids = self.reshape(input_ids, (bsz_per_core * self.num_labels, self.max_seq_length))
        input_mask = self.reshape(input_mask, (bsz_per_core * self.num_labels, self.max_seq_length))
        token_type_id = self.reshape(segment_ids, (bsz_per_core * self.num_labels, self.max_seq_length))
        _, logits, _ = self.albert(input_ids, input_mask, token_type_id)
        logits = self.reshape(logits, (bsz_per_core, self.num_labels))
        if not self.is_training:
            return logits
        log_probs = self.log_softmax(logits)
        loss = self.loss(log_probs, label_ids, self.num_labels)

        return loss
