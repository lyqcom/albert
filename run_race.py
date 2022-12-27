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
#################race albert example########################
python run_race.py
"""

import math
import os

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
from mindspore import log as logger
from mindspore import nn
from mindspore.communication.management import get_rank
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from dataset import create_race_dataset
from model_utils.config import get_config
from src import tokenization
from src.albert_for_finetune import AlbertRace
from src.assessment_method import Accuracy
from src.create_finetune_data import RaceProcessor
from src.utils import LossCallBack, AlbertLearningRate

args_opt = get_config("../task_race_config.yaml")
albert_net_cfg = args_opt.albert_net_cfg
optimizer_cfg = args_opt.optimizer_cfg
_cur_dir = os.getcwd()
print(args_opt)


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = args_opt.save_checkpoints_steps
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = AlbertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                         end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                         warmup_steps=args_opt.warmup_step,
                                         decay_steps=steps_per_epoch * epoch_num,
                                         power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = AlbertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                         end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                         warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                         decay_steps=steps_per_epoch * epoch_num,
                                         power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")
    # load network parameters
    if args_opt.load_last_finetune_checkpoint_path != '':
        param_dict = load_checkpoint(args_opt.load_last_finetune_checkpoint_path)
        load_param_into_net(network, param_dict)
    else:
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(network, param_dict)

    # config for saving checkpoint
    keep_checkpoint_max = math.ceil(dataset.get_dataset_size() * epoch_num / args_opt.save_checkpoints_steps)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoints_steps,
                                   keep_checkpoint_max=keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="race",
                                 directory=save_checkpoint_path,
                                 config=ckpt_config)
    # loss scaling
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    # network construction and update
    netwithgrads = TrainOneStepWithLossScaleCell(network, optimizer=optimizer, scale_sense=update_cell)
    # netwithgrads = AlbertRaceCell(network, optimizer=optimizer)

    logger.warning("train start…………")
    # do train
    # ----- 构造model的时候是否传入损失函数的参数？
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=False)


def do_eval(save_finetune_checkpoint_path, raceprocessor, tokenizer, label_list):
    """do evaluation"""
    import gc
    logger.warning("************ eval *************")
    if not os.path.exists(args_opt.eval_mindrecord_file):
        # 创建数据集
        # 一个问题一个example
        eval_examples = raceprocessor.get_test_examples(data_dir=args_opt.eval_dir_path)
        raceprocessor.convert_race_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=albert_net_cfg.seq_length,
            label_list=raceprocessor.get_labels(),
            max_qa_length=args_opt.max_qa_length,
            output_file=args_opt.eval_mindrecord_file)

    best_result = 0
    best_checkpoint = ''
    for filepath, _, filenames in os.walk(save_finetune_checkpoint_path):
        for index, checkpoint in enumerate(filenames):
            if not checkpoint.endswith("ckpt"):
                continue
            ds = create_race_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                     data_file_path=args_opt.eval_mindrecord_file,
                                     schema_file_path=args_opt.schema_file_path,
                                     do_shuffle=args_opt.eval_data_shuffle)
            net = AlbertRace(albert_net_cfg, False, len(label_list))
            net.set_train(False)
            print("number " + str(index) + " file " + checkpoint)
            checkpoint_path = os.path.join(filepath, checkpoint)
            param_dict = load_checkpoint(checkpoint_path)
            load_param_into_net(net, param_dict)
            eval_net = AlbertRaceEval(net)
            model = Model(eval_net, eval_network=eval_net, eval_indexes=None,
                          metrics={"Accuracy": Accuracy()})
            result = model.eval(ds)
            if result['Accuracy'] > best_result:
                best_result = result['Accuracy']
                if best_checkpoint != "":
                    os.remove(best_checkpoint)
                best_checkpoint = os.path.join(filepath, checkpoint)
                logger.warning("best result Accuracy:" + str(best_result))
            else:
                os.remove(checkpoint_path)
            del ds, net, param_dict, model, eval_net
            gc.collect()
    if not best_checkpoint.endswith("_best.ckpt"):
        os.rename(best_checkpoint, best_checkpoint.replace(".ckpt", "_best.ckpt"))
    print("==============================================================")
    print("race best eval :", best_result)
    print("==============================================================")
    with open(os.path.join(args_opt.load_finetune_checkpoint_path, "race_result.txt"), 'w') as wr:
        wr.write("accuracy {}".format(best_result))


class AlbertRaceEval(nn.Cell):
    def __init__(self, network):
        super(AlbertRaceEval, self).__init__()
        self.network = network

    def construct(self, input_ids, input_mask, token_type_id, label_id, is_real_example):
        logits = self.network(input_ids, input_mask, token_type_id, label_id, is_real_example)
        return logits, label_id


def run_race():
    """run squad task"""
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)
    if args_opt.distribute:
        if albert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            albert_net_cfg.compute_type = mstype.float32
        D.init()
        device_num = D.get_group_size()
        save_finetune_checkpoint_path = os.path.join(save_finetune_checkpoint_path, "distribute_ckpt")
        load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_path, "distribute_ckpt")
        save_finetune_checkpoint_path = os.path.join(save_finetune_checkpoint_path, str(get_rank()))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, device_num=device_num,
                                          parameter_broadcast=True, gradients_mean=True)
    else:
        save_finetune_checkpoint_path = os.path.join(save_finetune_checkpoint_path, "single_ckpt")
        load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_path, "single_ckpt")

    raceprocessor = RaceProcessor(do_lower_case=args_opt.do_lower_case,
                                  use_spm=args_opt.spm_model_file,
                                  high_only=args_opt.high_only,
                                  middle_only=args_opt.middle_only)

    label_list = raceprocessor.get_labels()

    # model construction
    netwithloss = AlbertRace(config=args_opt.albert_net_cfg,
                             is_training=True,
                             num_labels=len(label_list),
                             dropout_prob=0.1,
                             use_one_hot_embeddings=False)

    # train
    tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file_path,
                                           do_lower_case=True,
                                           spm_model_file=args_opt.spm_model_file)
    if args_opt.do_train:
        logger.warning("************ train *************")
        # 如果训练文件不存在就创建
        if not os.path.exists(args_opt.train_mindrecord_file):
            logger.warning("create mindrecord file……")
            train_examples = raceprocessor.get_train_examples(args_opt.train_dir_path)
            raceprocessor.convert_race_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=albert_net_cfg.seq_length,
                label_list=raceprocessor.get_labels(),
                max_qa_length=args_opt.max_qa_length,
                output_file=args_opt.train_mindrecord_file)
            logger.warning("create file over!!!")
        # 数据集创建
        ds = create_race_dataset(batch_size=args_opt.train_batch_size, repeat_count=1,
                                 data_file_path=args_opt.train_mindrecord_file,
                                 schema_file_path=args_opt.schema_file_path,
                                 do_shuffle=args_opt.train_data_shuffle)
        # 计算每一个所设置的step是多少个epoch
        args_opt.epoch_num = args_opt.epoch_num // ds.get_dataset_size()
        print("epoch_num:", args_opt.epoch_num)
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, args_opt.epoch_num)
        logger.warning("train over…………")

    # evaluation
    if args_opt.do_eval:
        do_eval(save_finetune_checkpoint_path, raceprocessor, tokenizer, label_list)


if __name__ == "__main__":
    run_race()
