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
#################squad v1.1 albert example########################
python run_squad_v1.py
"""

import collections
import os
import random
import json
import gc
import six


import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_rank
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from dataset import create_squad_dataset
from model_utils.config import get_config
from src import tokenization
from src.albert_for_finetune import AlbertSquad
from src.create_finetune_data import SquadV1Processor
from src.utils import LossCallBack, AlbertLearningRate

if six.PY2:
    import six.moves.cPickle as pickle
else:
    import pickle

args_opt = get_config("../task_squad_v1_config.yaml")
albert_net_cfg = args_opt.albert_net_cfg
optimizer_cfg = args_opt.optimizer_cfg
_cur_dir = os.getcwd()

print(args_opt)


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = AlbertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                         end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                         warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
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
    keep_checkpoint_max = args_opt.epoch_num * steps_per_epoch // steps_per_epoch
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                   keep_checkpoint_max=keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="squad",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    if args_opt.load_last_finetune_checkpoint_path != "":
        param_dict = load_checkpoint(args_opt.load_last_finetune_checkpoint_path)
        load_param_into_net(network, param_dict)
    else:
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    # netwithgrads = AlbertSquadCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads = TrainOneStepWithLossScaleCell(network, optimizer, update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=False)


def do_eval(save_finetune_checkpoint_path, squadprocessor, tokenizer):
    """do evaluation"""
    logger.warning("…………eval start…………")
    from src.squad_postprocess import accumulate_predictions_v1, write_predictions_v1, evaluate_v1

    eval_examples = squadprocessor.read_squad_examples(args_opt.eval_json_path, False)
    if not os.path.exists(args_opt.eval_mindrecord_file) or not os.path.exists(args_opt.eval_pkl_file):
        logger.warning("…………create eval Mindrecord file start…………")
        eval_features = squadprocessor.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=albert_net_cfg.seq_length,
            doc_stride=args_opt.doc_stride,
            max_query_length=args_opt.max_query_length,
            is_training=False,
            output_fn=args_opt.eval_mindrecord_file,
            do_lower_case=args_opt.do_lower_case)
        logger.warning("…………create eval Mindrecord file over…………")
        with open(args_opt.eval_pkl_file, "wb") as fout:
            pickle.dump(eval_features, fout)

    with open(args_opt.eval_pkl_file, "rb") as fin:
        eval_features = pickle.load(fin)
    with open(args_opt.eval_json_path) as predict_file:
        prediction_json = json.load(predict_file)["data"]

    def save_finetune_checkpoint():
        best_result_f1 = 0
        best_result_em = 0
        best_checkpoint = ''
        for filepath, _, filenames in os.walk(save_finetune_checkpoint_path):
            for index, checkpoint in enumerate(filenames):
                if not checkpoint.endswith("ckpt"):
                    continue
                ds = create_squad_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                          data_file_path=args_opt.eval_mindrecord_file,
                                          schema_file_path=args_opt.schema_file_path, is_training=False,
                                          do_shuffle=args_opt.eval_data_shuffle)
                net = AlbertSquad(albert_net_cfg, False, 2)
                net.set_train(False)
                print("number " + str(index) + " file " + checkpoint)
                checkpoint_path = os.path.join(filepath, checkpoint)
                param_dict = load_checkpoint(checkpoint_path)
                load_param_into_net(net, param_dict)
                model = Model(net)
                output = []
                RawResult = collections.namedtuple("RawResult", ["unique_id", "start_log_prob",
                                                                 "end_log_prob"])
                columns_list = ["input_ids", "input_mask", "segment_ids", "unique_id"]
                for data in ds.create_dict_iterator(num_epochs=1):
                    input_data = []
                    for i in columns_list:
                        input_data.append(data[i])
                    input_ids, input_mask, segment_ids, unique_ids = input_data
                    start_positions = Tensor([1], mstype.float32)
                    end_positions = Tensor([1], mstype.float32)
                    is_impossible = Tensor([1], mstype.float32)
                    logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                                           end_positions, unique_ids, is_impossible)
                    ids = logits[0].asnumpy()
                    start = logits[1].asnumpy()
                    end = logits[2].asnumpy()

                    for i in range(args_opt.eval_batch_size):
                        unique_id = int(ids[i])
                        start_logits = [float(x) for x in start[i].flat]
                        end_logits = [float(x) for x in end[i].flat]
                        output.append(RawResult(
                            unique_id=unique_id,
                            start_log_prob=start_logits,
                            end_log_prob=end_logits))
                output_prediction_file = os.path.join(
                    args_opt.output_path, "predictions.json")
                output_nbest_file = os.path.join(
                    args_opt.output_path, "nbest_predictions.json")
                result_dict = {}
                accumulate_predictions_v1(
                    result_dict, eval_examples, eval_features,
                    output, args_opt.n_best_size, args_opt.max_answer_length)
                predictions = write_predictions_v1(
                    result_dict, eval_examples, eval_features, output,
                    args_opt.n_best_size, args_opt.max_answer_length,
                    output_prediction_file, output_nbest_file)
                result = evaluate_v1(prediction_json, predictions)
                if result['f1'] > best_result_f1:
                    best_result_f1 = result['f1']
                    best_result_em = result['exact_match']
                    if best_checkpoint != "":
                        os.remove(best_checkpoint)
                    best_checkpoint = os.path.join(filepath, checkpoint)
                    logger.warning("best result f1/exact :" + str(best_result_f1) + "/" + str(best_result_em))
                else:
                    os.remove(checkpoint_path)
                del ds, net, param_dict, model, result_dict, output
                gc.collect()
        return best_result_f1, best_result_em, best_checkpoint
    best_result_f1, best_result_em, best_checkpoint = save_finetune_checkpoint()

    if not best_checkpoint.endswith("_best.ckpt"):
        os.rename(best_checkpoint, best_checkpoint.replace(".ckpt", "_best.ckpt"))
    print("==============================================================")
    print("squad_v1 best eval f1/em :", best_result_f1, best_result_em)
    print("==============================================================")
    save_path = os.path.join(save_finetune_checkpoint_path, "squad_v1_result.txt")
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            lines = f.readlines()
            args_opt.epoch_num += int(lines[-1].split(" ")[1])
    with open(save_path, 'a+') as wr:
        if os.path.exists(save_path):
            wr.write("\n")
        wr.write(
            "squad_v1 {} epoch eval result {} {}".format(str(args_opt.epoch_num), best_result_f1, best_result_em))


def run_squad():
    """run squad task"""
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)
    if args_opt.distribute:
        # if albert_net_cfg.compute_type != mstype.float32:
        #     logger.warning('GPU only support fp32 temporarily, run with fp32.')
        #     albert_net_cfg.compute_type = mstype.float32
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

    netwithloss = AlbertSquad(albert_net_cfg, True, 2, dropout_prob=0.1)

    squadprocessor = SquadV1Processor()

    tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file_path, do_lower_case=True,
                                           spm_model_file=args_opt.spm_model_file)

    if args_opt.do_train:
        logger.warning("…………train start…………")
        if not os.path.exists(args_opt.train_mindrecord_file):
            logger.warning("create train mindrecord file")
            # 一个问题一个example
            train_examples = squadprocessor.read_squad_examples(input_file=args_opt.train_json_path, is_training=True)

            rng = random.Random(12345)
            rng.shuffle(train_examples)

            squadprocessor.convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=albert_net_cfg.seq_length,
                doc_stride=args_opt.doc_stride,
                max_query_length=args_opt.max_query_length,
                is_training=True,
                output_fn=args_opt.train_mindrecord_file,
                do_lower_case=args_opt.do_lower_case)
            logger.warning("create mindrecord file over")

        ds = create_squad_dataset(batch_size=args_opt.train_batch_size, repeat_count=1,
                                  data_file_path=args_opt.train_mindrecord_file,
                                  schema_file_path=args_opt.schema_file_path,
                                  do_shuffle=args_opt.train_data_shuffle,
                                  is_v2=False)
        print("epcoh number :" + str(args_opt.epoch_num))
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, args_opt.epoch_num)

        logger.warning("…………train over…………")

    if args_opt.do_eval:
        do_eval(save_finetune_checkpoint_path, squadprocessor, tokenizer)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    run_squad()
