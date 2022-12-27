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
"""export checkpoint file into models"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export

from model_utils.config import get_config
from src.finetune_eval_model import AlbertCLSModel, AlbertSquadModel, AlbertSquadV2Model, AlbertRaceModel

args = get_config("../task_race_config.yaml")
albert_net_cfg = args.albert_net_cfg


def run_export():
    '''export function'''
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.description == "run_race":
        net = AlbertRaceModel(albert_net_cfg, False)
    elif args.description == "run_classifier":
        net = AlbertCLSModel(albert_net_cfg, False)
    elif args.description == "run_squad_v1":
        net = AlbertSquadModel(albert_net_cfg, False)
    elif args.description == "run_squad_v2":
        net = AlbertSquadV2Model(albert_net_cfg, False)
    else:
        raise ValueError("unsupported downstream task")

    load_checkpoint(args.export_ckpt_file, net=net)
    net.set_train(False)

    input_ids = Tensor(np.zeros([args.train_batch_size, albert_net_cfg.seq_length]), mstype.int32)
    input_mask = Tensor(np.zeros([args.train_batch_size, albert_net_cfg.seq_length]), mstype.int32)
    token_type_id = Tensor(np.zeros([args.train_batch_size, albert_net_cfg.seq_length]), mstype.int32)

    if args.description == "run_squad_v2":
        p_mask = Tensor(np.zeros([args.train_batch_size, albert_net_cfg.seq_length]), mstype.int32)
        start_positions = Tensor(np.zeros([args.train_batch_size, 1]), mstype.int32)
        input_data = [input_ids, input_mask, token_type_id, start_positions, p_mask]
    else:
        input_data = [input_ids, input_mask, token_type_id]
    export(net, *input_data, file_name=args.export_file_name, file_format=args.file_format)


if __name__ == "__main__":
    run_export()
