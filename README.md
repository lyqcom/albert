# 目录

[TOC]

# ALBERT概述

自BERT的成功以来，预训练模型都采用了很大的参数量以取得更好的模型表现。但是模型参数量越来越大也带来了很多问题，比如对算力要求越来越高、模型需要更长的时间去训练、甚至有些情况下参数量更大的模型表现却更差。
ALBERT是在BERT的基础上进行了三大改进：
    一. 嵌入向量参数化的因式分解
    二. 跨层参数共享（参数量减少主要贡献）
    三. 句间连贯性损失（SOP）
前两大改进解决了目前预训练模型参数量过大的问题，而第三大改进相比于NSP能够更好的学到句子间的连贯性

论文：[《ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS》](https://openreview.net/pdf?id=H1eA7AEtvS)

官方源码：[ALBERT 源码](https://github.com/google-research/albert)

# 模型架构

ALBERT的主干结构和BERT相同，都有三部分（Embedding、Encoder、Pooling）。但是ALBERT不直接将原本的one-hot向量映射到hidden space size of H，而是分解成两个矩阵，原本参数数量为V ∗ H，V表示的是Vocab Size。分解成两步则减少为V ∗ E + E ∗ H ，当H的值很大时，这样的做法能够大幅降低参数数量。

传统Transformer的每一层参数都是独立的，包括各层的self-attention、全连接。这样就导致层数增加时，参数量也会明显上升。ALBERT作者尝试将所有层的参数进行共享，相当于只学习第一层的参数，并在剩下的所有层中重用该层的参数，而不是每个层都学习不同的参数。

对于BERT_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块和前馈神经网络模块，每个自注意模块包含一个注意模块。对于BERT_NEZHA，Transformer包含24个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。BERT_base和BERT_NEZHA的区别在于，BERT_base使用绝对位置编码生成位置嵌入向量，而BERT_NEZHA使用相对位置编码。

# 数据集

* 生成预训练数据集
    * 下载[enwiki](https://dumps.wikimedia.org/enwiki/)数据集进行预训练
    * 使用WikiExtractor提取和整理数据集中的文本，使用步骤如下：
        * pip install wikiextractor
        * python -m wikiextractor.WikiExtractor -o -b
    * `WikiExtarctor`提取出来的原始文本并不能直接使用，还需要将数据集预处理并转换为 MindRecord格式。详见ALBERT代码仓中的create_pretraining_data.py文件，同时下载对应的vocab.txt文件, 如果出现AttributeError: module 'tokenization' has no attribute 'FullTokenizer’，请安装bert-tensorflow。
* 生成下游任务数据集
    * 下载数据集进行微调和评估，如RACE、SQuAD v1.1、SQuAD v2.0等。
    * 将数据集文件从JSON格式转换为MindRecord格式。详见ALBERT代码仓中的run_race.py、run_squad_v1.py或run_squad_v2.py文件。

# 环境要求

* **硬件**
    * 准备GPU处理器搭建硬件环境。
* **框架**
    * [Mindspore框架](https://gitee.com/mindspore/mindspore)
* 更多关于Mindspore的信息，请查看以下资源：
    * [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    * [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

* 在Gpu上运行

  ```sh
  # 单机运行预训练示例
  run_pretrain_single.sh
  # 分布式运行预训练示例
  run_pretrain_dis.sh
  # 运行微调和评估示例
  - 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
  - 在`task_[DOWNSTREAM_TASK]_config.yaml`中设置ALBERT网络配置和优化器超参。
  -----------------------------------------------------
  - RACE任务：在scripts/run_race_single.sh中设置任务相关的超参。
  - 运行`bash scripts/run_race_single.sh`，单机下对ALBERT-base模型进行微调。
    bash scripts/run_race_single.sh
  -----------------------------------------------------
  - SQUAD v1.1任务：在scripts/run_squad_v1_single.sh中设置任务相关的超参。
  -运行`bash scripts/run_squad_v1_single.sh`，单机下对ALBERT-base模型进行微调。
    bash scripts/run_squad_v1_single.sh
  -----------------------------------------------------
  - SQUAD v2.0任务：在scripts/run_squad_v2_single.sh中设置任务相关的超参。
  -运行`bash scripts/run_squad_v2_single.sh`，单机下对ALBERT-base模型进行微调。
    bash scripts/run_squad_v2_single.sh
  ```

## 脚本说明

## 脚本和样例代码

```python
.
└─albert
    ├─README.md
    ├─model_utils
        └─config.py                                     # 解析*.yaml参数配置文件
    ├─scripts
        ├─run_classifier_distribute.sh                  # GPU设备上分布式classifier任务shell脚本
        ├─run_classifier_single.sh                      # GPU设备上单机classifier任务shell脚本
        ├─run_pretrain_distribute.sh                    # GPU设备上分布式预训练任务shell脚本
        ├─run_pretrain_single.sh                        # GPU设备上单机预训练任务shell脚本
        ├─run_race_distribute.sh                        # GPU设备上分布式RACE任务shell脚本
        ├─run_race_single.sh                            # GPU设备上单机RACE任务shell脚本
        ├─run_squad_v1_distribute.sh                    # GPU设备上分布式SQuAD v1.1任务shell脚本
        ├─run_squad_v1_single.sh                        # GPU设备上单机SQuAD v1.1任务shell脚本
        ├─run_squad_v2_distribute.sh                    # GPU设备上分布式SQuAD v2.0任务shell脚本
        └─run_squad_v2_single.sh                        # GPU设备上单机SQuAD v2.0任务shell脚本
    ├─src
        ├─albert_for_finetune.py                        # 网络骨干编码
        ├─albert_for_pre_training.py                    # 网络骨干编码
        ├─albert_model.py                               # 网络骨干编码
        ├─assessment_method.py                          # 评估过程的测评方法
        ├─create_finetune_data.py                       # 创建微调（classifier和RACE任务）数据
        ├─create_pretraining_data.py                    # 创建预训练数据
        ├─create_squad_data.py                          # 创建SQuAD任务数据
        ├─finetune_eval_model.py                        # 网络骨干编码
        ├─squad_get_predictions.py                      # 得到SQuAD任务的评估结果
        ├─squad_postprocess.py                          # 得到SQuAD任务的评估结果
        ├─tokenization.py                               # 文本处理
        └─utils.py                                      # util函数
    ├─create_data_config.yaml                           # 创建数据集的参数配置
    ├─dataset.py                                        # 加载数据集脚本
    ├─export.py                                         # 导出checkpoint文件
    ├─pretrain_config.yaml                              # 预训练参数配置
    ├─pretrain_eval.py                                  # 训练和评估网络
    ├─requirements.txt                                  # 环境要求
    ├─run_classifier.py                                 # classifier任务的微调和评估网络
    ├─run_pretrain.py                                   # 预训练网络
    ├─run_race.py                                       # RACE任务的微调和评估网络
    ├─run_squad_v1.py                                   # SQuAD v1.1任务的微调和评估网络
    ├─run_squad_v2.py                                   # SQuAD v2.0任务的微调和评估网络
    ├─task_classifier_config.yaml                       # 下游任务classifier参数配置
    ├─task_race_config.yaml                             # 下游任务RACE参数配置
    ├─task_squad_v1_config.yaml                         # 下游任务SQuAD v1.1参数配置
    └─task_squad_v2_config.yaml                         # 下游任务SQuAD v2.0参数配置
```

## 脚本参数

### 预训练

```python
用法：run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

选项：
    --device_target            代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --distribute               是否多卡预训练，可选项为true（多卡预训练）或false。默认为false
    --epoch_size               轮次，默认为1
    --device_num               使用设备数量，默认为1
    --device_id                设备ID，默认为0
    --enable_save_ckpt         是否使能保存检查点，可选项为true或false，默认为true
    --enable_lossscale         是否使能损失放大，可选项为true或false，默认为true
    --do_shuffle               是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink         是否使能数据下沉，可选项为true或false，默认为true
    --data_sink_steps          设置数据下沉步数，默认为1
    --accumulation_steps       更新权重前梯度累加数，默认为1
    --save_checkpoint_path     保存检查点文件的路径，默认为""
    --load_checkpoint_path     加载检查点文件的路径，默认为""
    --save_checkpoint_steps    保存检查点文件的步数，默认为1000
    --save_checkpoint_num      保存的检查点文件数量，默认为1
    --train_steps              训练步数，默认为-1
    --data_dir                 数据目录，默认为""
    --schema_dir               schema.json的路径，默认为""
```

### 微调与评估

```python
用法: run_race.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                      [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [-num_class N]
                      [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                      [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                      [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                      [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                      [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                      [--train_data_file_path TRAIN_DATA_FILE_PATH]
                      [--eval_data_file_path EVAL_DATA_FILE_PATH]
                      [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   任务运行的目标设备，可选项为Ascend或GPU
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为accuracy、f1、mcc、spearman_correlation
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       标注类的数量
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练ALBERT模型）
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             用于保存预测数据的TFRecord文件，如dev.tfrecord
    --schema_file_path                模式文件保存路径

用法：run_squad_v1.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       分类数，SQuAD任务通常为2
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 ALBERT模型训练的词汇表
    --eval_json_path                  保存SQuAD任务开发JSON文件的路径
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练ALBERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存SQuAD训练数据的TFRecord文件，如train1.1.tfrecord
    --eval_data_file_path             用于保存SQuAD预测数据的TFRecord文件，如dev1.1.tfrecord
    --schema_file_path                模式文件保存路径

用法：run_squad_v2.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       分类数，SQuAD任务通常为2
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 ALBERT模型训练的词汇表
    --eval_json_path                  保存SQuAD任务开发JSON文件的路径
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练ALBERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存SQuAD训练数据的TFRecord文件，如train1.1.tfrecord
    --eval_data_file_path             用于保存SQuAD预测数据的TFRecord文件，如dev1.1.tfrecord
    --schema_file_path                模式文件保存路径
```

## 选项及参数

可以在yaml配置文件中分别配置预训练和下游任务的参数。

### 选项

```python
config for lossscale and etc.
    albert_network                  ALBERT模型版本，可选项为albert_base、albert_large、albert_xlarge、albert_xxlarge，默认为base
    batch_size                      输入数据集的批次大小，默认为32
    loss_scale_value                损失放大初始值，默认为2^32
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为1000
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecay和Lamb，默认为AdamWerigtDecay
```

### 参数

```python
数据集和网络参数（预训练/微调/评估）：
    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为21136
    hidden_size                     AlBERT的encoder层数，默认为768
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数，默认为3072
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             ALBERT输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    ALBERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    ALBert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为30000
    hidden_size                     AlBERT的encoder层数，默认为768
    embedding_size                  因式分解的大小，默认为128
    num_hidden_groups               共享相同参数的隐藏层组数， 默认为1
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               隐藏层维度，默认为3072
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             ALBERT输出的随机失活可能性，默认为0.1
    inner_group_num
    attention_probs_dropout_prob    ALBERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为True
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    ALBert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

    Lamb:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
```

## 训练过程

### 用法

#### GPU处理器上运行

```bash
bash scripts/run_classifier_single.sh                   # 可自行修改脚本里面的参数配置
```

以上命令后台运行，您可以在./output/logs/pre_train_s.log中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

### 分布式训练

#### GPU处理器上运行

```bash
bash scripts/run_classifier_distribute.sh             # 可自行修改脚本里面的参数配置
```

以上命令后台运行，您可以在./output/logs/pre_train_d.log中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

## 评估过程

### 用法

#### GPU处理器上运行后评估RACE数据集

运行以下命令前，确保已设置加载与训练检查点路径，请将检查点路径设置为绝对全路径。

```bash
bash scripts/run_classifier_single.sh
```

以上命令后台运行，您可以在./output/logs/classifier_log.log中查看训练日志。

如您选择准确性作为评估方法，可得到如下结果：

```python
epoch: 0, current epoch percent: 0.001, step: 1, outputs are (Tensor(shape=[], dtype=Float32, value= 1.37603), Tensor(shape=[], dtype=Bool, value= True), Parameter (name=scale_sense, shape=(), dtype=Float32, requires_grad=True))
epoch: 0, current epoch percent: 0.002, step: 2, outputs are (Tensor(shape=[], dtype=Float32, value= 1.36489), Tensor(shape=[], dtype=Bool, value= True), Parameter (name=scale_sense, shape=(), dtype=Float32, requires_grad=True))
......
```

#### GPU处理器上运行后评估SQuAD v1.1数据集

运行以下命令前，确保已设置加载与训练检查点路径，请将检查点路径设置为绝对全路径。

```bash
bash scripts/run_squad_v1_single.sh
```

以上命令后台运行，您可以在./output/logs/squad_v1.log中查看训练日志。

如您选择准确性作为评估方法，可得到如下结果：

```python
epoch: 0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[], dtype=Float32, value= 5.91539), Tensor(shape=[], dtype=Bool, value= False), Parameter (name=scale_sense, shape=(), dtype=Float32, requires_grad=True))
epoch: 0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[], dtype=Float32, value= 5.9899), Tensor(shape=[], dtype=Bool, value= False), Parameter (name=scale_sense, shape=(), dtype=Float32, requires_grad=True))
......
```

#### GPU处理器上运行后评估SQuAD v2.0数据集

运行以下命令前，确保已设置加载与训练检查点路径，请将检查点路径设置为绝对全路径。

```bash
bash scripts/run_squad_v2_single.sh
```

以上命令后台运行，您可以在./output/logs/squad_v2.log 中查看训练日志。

如您选择准确性作为评估方法，可得到如下结果：

```bash
epoch: 0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[], dtype=Float32, value= 5.28869), Tensor(shape=[], dtype=Bool, value= False), Parameter (name=scale_sense, shape=(), dtype=Float32, requires_grad=True))
epoch: 0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[], dtype=Float32, value= 5.38487), Tensor(shape=[], dtype=Bool, value= False), Parameter (name=scale_sense, shape=(), dtype=Float32, requires_grad=True))
......
```

## 导出mindir模型

由于预训练模型通常没有应用场景，需要经过下游任务的finetune之后才能使用，所以当前仅支持使用下游任务模型和yaml配置文件进行export操作。

```bash
python export.py --config_path [/path/*.yaml] --export_ckpt_file [CKPT_PATH] --export_file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`export_ckpt_file` 是必需的，`file_format` 必须在 ["AIR", "MINDIR"]中进行选择。

## 模型描述

## 性能

### 预训练性能

| 参数          | GPU                  |
| ------------- | -------------------- |
| 模型版本      | ALBERT_base          |
| 资源          | A100                 |
| 上传日期      | 2021-12-19           |
| MindSpore版本 | 1.5.0                |
| 数据集        | en-wiki              |
| 训练参数      | pretrain_config.yaml |
| 优化器        | AdamWeightDecay      |
| 损失函数      | SoftmaxCrossEntropy  |
| 输出          | 概率                 |
| 轮次          | 80                   |
| Batch_size    | 96                   |
| 损失          | 10.972               |
| 速度          | 1180.973毫秒/步      |
| 总时长        | 82小时               |
| 参数（M）     | 45.069               |
| 微调检查点    |                      |

# 随机情况说明

训练与微调中，设置train_data_shuffle和eval_data_shuffle为True，则对数据集进行轮换操作。

配置文件*.yaml中，若将hidden_dropout_prob和attention_probs_dropout_prob设置为非0，则在训练过程中丢弃部分网络节点。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ)来查找一些常见的公共问题。

