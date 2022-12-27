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
#################pre_train albert example on zh-wiki########################
python run_pretrain.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
from mindspore import log as logger
from mindspore.common import set_seed
from mindspore.communication.management import get_rank
from mindspore.context import ParallelMode
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore.nn.optim import Lamb, AdamWeightDecay
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from dataset import create_albert_dataset
from model_utils.config import get_config
from src.albert_for_pre_training import AlbertNetworkWithLoss
from src.utils import LossCallBack, AlbertLearningRate

_current_dir = os.path.dirname(os.path.realpath(__file__))
cfg = get_config('../pretrain_config.yaml')
albert_net_cfg = cfg.albert_net_cfg


def _set_albert_all_reduce_split():
    """set bert all_reduce fusion split, support num_hidden_layers is 12 and 24."""
    device_target = context.get_context('device_target')
    enable_graph_kernel = context.get_context('enable_graph_kernel')
    device_num = context.get_auto_parallel_context('device_num')
    if albert_net_cfg.num_hidden_layers == 12:
        if albert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 87, 116, 145, 174, 203, 217])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[28, 55, 82, 109, 136, 163, 190, 205])
            if device_target == 'GPU' and enable_graph_kernel and device_num == 8:
                context.set_auto_parallel_context(all_reduce_fusion_config=[180, 205])
            elif device_target == 'GPU' and enable_graph_kernel and device_num == 16:
                context.set_auto_parallel_context(all_reduce_fusion_config=[120, 205])
    elif albert_net_cfg.num_hidden_layers == 24:
        if albert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[30, 90, 150, 210, 270, 330, 390, 421])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[38, 93, 148, 203, 258, 313, 368, 397])
            if device_target == 'Ascend' and enable_graph_kernel and device_num == 8:
                context.set_auto_parallel_context(all_reduce_fusion_config=[
                    0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 93, 148, 203, 258, 313, 368, 397])


def _get_optimizer(args_opt, network):
    """get bert optimizer, support Lamb, Momentum, AdamWeightDecay."""
    if cfg.optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = AlbertLearningRate(learning_rate=cfg.optimizer_cfg.Lamb.learning_rate,
                                         end_learning_rate=cfg.optimizer_cfg.Lamb.end_learning_rate,
                                         warmup_steps=cfg.optimizer_cfg.Lamb.warmup_steps,
                                         decay_steps=args_opt.train_steps,
                                         power=cfg.optimizer_cfg.Lamb.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.optimizer_cfg.Lamb.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.optimizer_cfg.Lamb.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.optimizer_cfg.Lamb.optimizer_cfg.weight_decay},
                        {'params': other_params},
                        {'order_params': params}]
        optimizer = Lamb(group_params, beta1=0.9, beta2=0.999, learning_rate=lr_schedule, eps=cfg.Lamb.eps)
    elif cfg.optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = AlbertLearningRate(learning_rate=cfg.optimizer_cfg.AdamWeightDecay.learning_rate,
                                         end_learning_rate=cfg.optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                         warmup_steps=cfg.optimizer_cfg.AdamWeightDecay.warmup_steps,
                                         decay_steps=args_opt.train_steps,
                                         power=cfg.optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=cfg.optimizer_cfg.AdamWeightDecay.eps)
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, AdamWeightDecay, Thor]".
                         format(cfg.optimizer))
    return optimizer


def _set_graph_kernel_context(device_target):
    """Add suitable graph kernel context for different configs."""
    if device_target == 'GPU':
        # 启用图核融合以优化网络执行性能。
        if cfg.albert_network == 'base':
            context.set_context(enable_graph_kernel=True,
                                graph_kernel_flags="--enable_stitch_fusion=true "
                                                   "--enable_parallel_fusion=true "
                                                   "--enable_cluster_ops=BatchMatMul")
        else:
            context.set_context(enable_graph_kernel=True)
    else:
        logger.warning('Graph kernel only supports GPU back-end now, run with graph kernel off.')


def _check_compute_type(args_opt):
    if args_opt.device_target == 'GPU' and albert_net_cfg.compute_type \
            != mstype.float32 and cfg.albert_network != 'base':
        warning_message = 'Gpu only support fp32 temporarily, run with fp32.'
        albert_net_cfg.compute_type = mstype.float32
        if args_opt.enable_lossscale == "true":
            args_opt.enable_lossscale = "false"
            warning_message = 'Gpu only support fp32 temporarily, run with fp32 and disable lossscale.'
        logger.warning(warning_message)


def run_pretrain():
    """pre-train bert_clue"""
    context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.device_target)
    context.set_context(reserve_class_name_in_scope=False)
    ckpt_save_dir = cfg.save_checkpoint_path
    if cfg.distribute:
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        ckpt_save_dir = os.path.join(cfg.save_checkpoint_path, 'ckpt_' + str(get_rank()))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        # _set_albert_all_reduce_split()
    else:
        rank = 0
        device_num = 1
    _check_compute_type(cfg)
    if cfg.accumulation_steps > 1:
        # logger.info("accumulation steps: {}".format(cfg.accumulation_steps))
        # logger.info("global batch size: {}".format(cfg.batch_size * cfg.accumulation_steps))
        # 是否使能数据下沉，默认为true
        if cfg.enable_data_sink:
            cfg.data_sink_steps *= cfg.accumulation_steps
            logger.info("data sink steps: {}".format(cfg.data_sink_steps))
        # 是否使能保存检查点，默认为true
        if cfg.enable_save_ckpt:
            cfg.save_checkpoint_steps *= cfg.accumulation_steps
            logger.info("save checkpoint steps: {}".format(cfg.save_checkpoint_steps))
    ds = create_albert_dataset(device_num, rank, cfg.do_shuffle, cfg.data_dir, cfg.train_batch_size)

    net_with_loss = AlbertNetworkWithLoss(albert_net_cfg, True)

    optimizer = _get_optimizer(cfg, net_with_loss)

    callback = [TimeMonitor(cfg.data_sink_steps), LossCallBack(ds.get_dataset_size())]
    if cfg.enable_save_ckpt and cfg.device_id % min(8, device_num) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                     keep_checkpoint_max=cfg.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_albert',
                                     directory=None if ckpt_save_dir == "" else ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)
    if cfg.load_checkpoint_path:
        param_dict = load_checkpoint(cfg.load_checkpoint_path)
        load_param_into_net(net_with_loss, param_dict)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                             scale_factor=cfg.scale_factor,
                                             scale_window=cfg.scale_window)
    net_with_grads = TrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer, scale_sense=update_cell)
    model = Model(net_with_grads)
    model.train(cfg.epoch, ds, callbacks=callback,
                dataset_sink_mode=cfg.enable_data_sink,
                sink_size=cfg.data_sink_steps)


if __name__ == '__main__':
    set_seed(0)
    run_pretrain()
