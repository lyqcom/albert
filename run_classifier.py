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
Albert finetune and evaluation script.
'''
import math
import os

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore.nn as nn
from mindspore import context
from mindspore import log as logger
from mindspore.communication.management import get_rank
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from dataset import create_classification_dataset
from model_utils.config import get_config
import src.create_finetune_data as data_utils
from src import tokenization
from src.albert_for_finetune import AlbertCLS
from src.albert_for_finetune import AlbertCLSModel
from src.assessment_method import Streaming_Pearson_Correlation, CLSMetric
from src.utils import LossCallBack, AlbertLearningRate

args_opt = get_config("../task_classifier_config.yaml")
albert_net_cfg = args_opt.albert_net_cfg
optimizer_cfg = args_opt.optimizer_cfg
_cur_dir = os.getcwd()
print(args_opt)


def do_train(dataset=None, network=None, load_checkpoint_path="",
             save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
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
    # 计算保存的文件数量,每隔save_checkpoint_steps保存一个文件
    keep_checkpoint_max = math.ceil(dataset.get_dataset_size() * epoch_num / args_opt.save_checkpoint_steps)
    print("save checkpoint number ：" + str(keep_checkpoint_max))
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                   keep_checkpoint_max=keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="classifier",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    # 加载上次训练的checkpoint
    if args_opt.load_last_finetune_checkpoint_path != "":
        param_dict = load_checkpoint(args_opt.load_last_finetune_checkpoint_path)
        load_param_into_net(network, param_dict)
    else:
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)

    netwithgrads = TrainOneStepWithLossScaleCell(network, optimizer, update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=False)


def do_eval(save_finetune_checkpoint_path, processor, tokenizer, label_list):
    """do evaluation"""
    import gc
    logger.warning("************ eval *************")
    if not os.path.exists(args_opt.eval_mindrecord_file):
        eval_examples = processor.get_dev_examples(args_opt.eval_data_file_path)
        data_utils.convert_classifier_examples_to_features(
            examples=eval_examples,
            label_list=processor.get_labels(),
            max_seq_length=albert_net_cfg.seq_length,
            tokenizer=tokenizer,
            task_name=args_opt.task_name.lower(),
            vocab_file=args_opt.vocab_file_path,
            output_file=args_opt.eval_mindrecord_file)
    best_result = 0
    best_checkpoint = ''
    # 从保存的所有checkpoint文件中选取最好的结果
    for filepath, _, filenames in os.walk(save_finetune_checkpoint_path):
        for index, checkpoint in enumerate(filenames):
            if not checkpoint.endswith("ckpt"):
                continue
            ds = create_classification_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                               data_path=args_opt.eval_mindrecord_file,
                                               task_name=args_opt.task_name.lower(),
                                               do_shuffle=args_opt.eval_data_shuffle, is_training=False)

            net_for_pretraining = AlbertCLSModel(albert_net_cfg, False, len(label_list),
                                                 task_name=args_opt.task_name.lower())
            net_for_pretraining.set_train(False)
            print("number " + str(index) + " file " + checkpoint)
            checkpoint_path = os.path.join(filepath, checkpoint)
            param_dict = load_checkpoint(checkpoint_path)
            load_param_into_net(net_for_pretraining, param_dict)
            # 定义评估网络
            eval_net = AlbertCLSEval(net_for_pretraining)
            if args_opt.task_name.lower() == "sts-b":
                eval_method = Streaming_Pearson_Correlation()
            else:
                eval_method = CLSMetric()
            model = Model(eval_net, eval_network=eval_net, eval_indexes=None,
                          metrics={"Accuracy": eval_method})
            # result=do_eval(ds, len(label_list), checkpoint_path,args_opt.task_name.lower())
            result = model.eval(ds)
            print("current result " + str(result['Accuracy']))
            if result['Accuracy'] > best_result:
                best_result = result['Accuracy']
                if best_checkpoint != "":
                    os.remove(best_checkpoint)
                best_checkpoint = os.path.join(filepath, checkpoint)
                logger.warning("best result Accuracy:" + str(best_result))
            else:
                os.remove(checkpoint_path)
            del ds, net_for_pretraining, param_dict, model, eval_net
            gc.collect()
    if not best_checkpoint.endswith("_best.ckpt"):
        os.rename(best_checkpoint, best_checkpoint.replace(".ckpt", "_best.ckpt"))
    print("==============================================================")
    print("{} accuracy {}".format(args_opt.task_name, best_result))
    print("==============================================================")
    save_path = os.path.join(args_opt.save_finetune_checkpoint_path, args_opt.task_name + "_result.txt")
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            lines = f.readlines()
            args_opt.epoch_num += int(lines[-1].split(" ")[1])
    with open(save_path, 'a+') as wr:
        if os.path.exists(save_path):
            wr.write("\n")
        wr.write("{} {} epoch accuracy {}".format(args_opt.task_name.upper(), str(args_opt.epoch_num), best_result))
class AlbertCLSEval(nn.Cell):

    def __init__(self, network):
        super(AlbertCLSEval, self).__init__()
        self.albert = network

    def construct(self, input_ids, input_mask, token_type_id, label_id, is_real_example):
        logits = self.albert(input_ids, input_mask, token_type_id)
        return logits, label_id


def run_classifier():
    """run classifier task"""

    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = os.path.join(args_opt.save_finetune_checkpoint_path, args_opt.task_name)
    load_finetune_checkpoint_path = os.path.join(args_opt.load_finetune_checkpoint_path, args_opt.task_name)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
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
                                          parameter_broadcast=True)
    else:
        save_finetune_checkpoint_path = os.path.join(save_finetune_checkpoint_path, "single_ckpt")
        load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_path, "single_ckpt")
    # 判断是调用哪一个分类任务
    processors = {
        "cola": data_utils.ColaProcessor,
        "mnli": data_utils.MnliProcessor,
        "mismnli": data_utils.MisMnliProcessor,
        "mrpc": data_utils.MrpcProcessor,
        "rte": data_utils.RteProcessor,
        "sst-2": data_utils.Sst2Processor,
        "sts-b": data_utils.StsbProcessor,
        "qqp": data_utils.QqpProcessor,
        "qnli": data_utils.QnliProcessor,
        "wnli": data_utils.WnliProcessor,
    }
    if args_opt.task_name.lower() in processors.keys():
        processor = processors[args_opt.task_name.lower()](use_spm=args_opt.spm_model_file,
                                                           do_lower_case=True)
        args_opt.train_mindrecord_file = args_opt.train_mindrecord_file + \
                                         args_opt.task_name.lower() + \
                                         "_train.mindrecord"
        args_opt.eval_mindrecord_file = args_opt.eval_mindrecord_file + \
                                        args_opt.task_name.lower() + \
                                        "_eval.mindrecord"
    else:
        raise Exception("this classify task isn't supported")
    label_list = processor.get_labels()
    netwithloss = AlbertCLS(albert_net_cfg, True,
                            num_labels=len(label_list), dropout_prob=0.1, task_name=args_opt.task_name.lower())

    tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file_path, do_lower_case=True,
                                           spm_model_file=args_opt.spm_model_file)
    if args_opt.do_train:
        logger.warning("************ train *************")
        if not os.path.exists(args_opt.train_mindrecord_file):
            logger.warning("create mindrecord file……")
            train_examples = processor.get_train_examples(args_opt.train_data_file_path)
            data_utils.convert_classifier_examples_to_features(
                examples=train_examples,
                label_list=processor.get_labels(),
                max_seq_length=albert_net_cfg.seq_length,
                tokenizer=tokenizer,
                task_name=args_opt.task_name.lower(),
                vocab_file=args_opt.vocab_file_path,
                output_file=args_opt.train_mindrecord_file)
            logger.warning("create file over!!!")
        ds = create_classification_dataset(batch_size=args_opt.train_batch_size, repeat_count=1,
                                           data_path=args_opt.train_mindrecord_file,
                                           task_name=args_opt.task_name.lower(),
                                           do_shuffle=args_opt.train_data_shuffle)
        args_opt.epoch_num = args_opt.epoch_num // ds.get_dataset_size()
        print("epcoh number :" + str(args_opt.epoch_num))
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, args_opt.epoch_num)
        logger.warning("train over…………")

    if args_opt.do_eval:
        do_eval(save_finetune_checkpoint_path, processor, tokenizer, label_list)


if __name__ == "__main__":
    run_classifier()
