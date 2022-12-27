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

"""create fintune data"""
import collections
import csv
import json
import os
import six
import numpy as np

from mindspore.mindrecord import FileWriter
from src import tokenization


class SquadExample:
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeaturesSquad:
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tok_start_to_orig_index,
                 tok_end_to_orig_index,
                 token_is_max_context,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 paragraph_len,
                 p_mask=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.p_mask = p_mask


class SquadV1Processor:
    """processor for suqad v1.1"""
    def read_squad_examples(self, input_file, is_training):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    orig_answer_text = None
                    is_impossible = False

                    if is_training:
                        is_impossible = qa.get("is_impossible", False)
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            start_position = answer["answer_start"]
                        else:
                            start_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        paragraph_text=paragraph_text,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        return examples

    def _convert_index(self, index, pos, m=None, is_start=True):
        """Converts index."""
        if index[pos] is not None:
            return index[pos]
        n = len(index)
        rear = pos
        while rear < n - 1 and index[rear] is None:
            rear += 1
        front = pos
        while front > 0 and index[front] is None:
            front -= 1
        assert index[front] is not None or index[rear] is not None
        if index[front] is None:
            if index[rear] >= 1:
                if is_start:
                    return 0
                return index[rear] - 1
            return index[rear]
        if index[rear] is None:
            if m is not None and index[front] < m - 1:
                if is_start:
                    return index[front] + 1
                return m - 1
            return index[front]
        if is_start:
            if index[rear] > index[front] + 1:
                return index[front] + 1
            return index[rear]

        if index[rear] > index[front] + 1:
            return index[rear] - 1
        return index[front]

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length,
                                     doc_stride, max_query_length, is_training,
                                     output_fn, do_lower_case):
        """Loads a data file into a list of `InputBatch`s."""

        cnt_pos, cnt_neg = 0, 0
        unique_id = 1000000000
        max_n, max_m = 1024, 1024
        f = np.zeros((max_n, max_m), dtype=np.float32)
        writer = FileWriter(file_name=output_fn, shard_num=1)
        data_schema = {"unique_id": {"type": "int64", "shape": [-1]},
                       "input_ids": {"type": "int64", "shape": [-1]},
                       "input_mask": {"type": "int64", "shape": [-1]},
                       "segment_ids": {"type": "int64", "shape": [-1]},
                       "p_mask": {"type": "int64", "shape": [-1]},
                       }
        if is_training:
            data_schema["start_position"] = {"type": "int64", "shape": [-1]}
            data_schema["end_position"] = {"type": "int64", "shape": [-1]}
            data_schema["is_impossible"] = {"type": "int64", "shape": [-1]}
        writer.add_schema(data_schema)
        data_input = []
        output = []
        g = {}
        def _lcs_match(max_dist, n, m):
            """Longest-common-substring algorithm."""
            f.fill(0)
            g.clear()

            # longest common sub sequence
            # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
            for i in range(n):

                # note(zhiliny):
                # unlike standard LCS, this is specifically optimized for the setting
                # because the mismatch between sentence pieces and original text will
                # be small
                for j in range(i - max_dist, i + max_dist):
                    if j >= m or j < 0: continue

                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    if (tokenization.preprocess_text(
                            paragraph_text[i], lower=do_lower_case,
                            remove_space=False) == tok_cat_text[j]
                            and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        for (example_index, example) in enumerate(examples):

            if example_index % 100 == 0:
                print("Converting {}/{} pos {} neg {}".format(
                    example_index, len(examples), cnt_pos, cnt_neg))

            query_tokens = tokenization.encode_ids(
                tokenizer.sp_model,
                tokenization.preprocess_text(
                    example.question_text, lower=do_lower_case))

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            paragraph_text = example.paragraph_text
            para_tokens = tokenization.encode_pieces(
                tokenizer.sp_model,
                tokenization.preprocess_text(
                    example.paragraph_text, lower=do_lower_case),
                return_unicode=False)

            chartok_to_tok_index = []
            tok_start_to_chartok_index = []
            tok_end_to_chartok_index = []
            char_cnt = 0
            para_tokens = [six.ensure_text(token, "utf-8") for token in para_tokens]
            for i, token in enumerate(para_tokens):
                new_token = six.ensure_text(token).replace(
                    tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
                chartok_to_tok_index.extend([i] * len(new_token))
                tok_start_to_chartok_index.append(char_cnt)
                char_cnt += len(new_token)
                tok_end_to_chartok_index.append(char_cnt - 1)

            tok_cat_text = "".join(para_tokens).replace(
                tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
            n, m = len(paragraph_text), len(tok_cat_text)

            if n > max_n or m > max_m:
                max_n = max(n, max_n)
                max_m = max(m, max_m)
                f = np.zeros((max_n, max_m), dtype=np.float32)

            max_dist = abs(n - m) + 5
            for _ in range(2):
                _lcs_match(max_dist, n, m)
                if f[n - 1, m - 1] > 0.8 * n: break
                max_dist *= 2

            orig_to_chartok_index = [None] * n
            chartok_to_orig_index = [None] * m
            i, j = n - 1, m - 1
            while i >= 0 and j >= 0:
                if (i, j) not in g: break
                if g[(i, j)] == 2:
                    orig_to_chartok_index[i] = j
                    chartok_to_orig_index[j] = i
                    i, j = i - 1, j - 1
                elif g[(i, j)] == 1:
                    j = j - 1
                else:
                    i = i - 1

            if (all(v is None for v in orig_to_chartok_index) or
                    f[n - 1, m - 1] < 0.8 * n):
                print("MISMATCH DETECTED!")
                continue

            tok_start_to_orig_index = []
            tok_end_to_orig_index = []
            for i in range(len(para_tokens)):
                start_chartok_pos = tok_start_to_chartok_index[i]
                end_chartok_pos = tok_end_to_chartok_index[i]
                start_orig_pos = self._convert_index(chartok_to_orig_index, start_chartok_pos,
                                                     n, is_start=True)
                end_orig_pos = self._convert_index(chartok_to_orig_index, end_chartok_pos,
                                                   n, is_start=False)

                tok_start_to_orig_index.append(start_orig_pos)
                tok_end_to_orig_index.append(end_orig_pos)

            if not is_training:
                tok_start_position = tok_end_position = None

            if is_training and example.is_impossible:
                tok_start_position = 0
                tok_end_position = 0

            if is_training and not example.is_impossible:
                start_position = example.start_position
                end_position = start_position + len(example.orig_answer_text) - 1

                start_chartok_pos = self._convert_index(orig_to_chartok_index, start_position,
                                                        is_start=True)
                tok_start_position = chartok_to_tok_index[start_chartok_pos]

                end_chartok_pos = self._convert_index(orig_to_chartok_index, end_position,
                                                      is_start=False)
                tok_end_position = chartok_to_tok_index[end_chartok_pos]
                assert tok_start_position <= tok_end_position

            def _piece_to_id(x):
                if six.PY2 and isinstance(x, six.text_type):
                    x = six.ensure_binary(x, "utf-8")
                return tokenizer.sp_model.PieceToId(x)

            all_doc_tokens = list(map(_piece_to_id, para_tokens))

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                ep = {}
                tokens = []
                token_is_max_context = {}
                segment_ids = []
                p_mask = []

                cur_tok_start_to_orig_index = []
                cur_tok_end_to_orig_index = []

                tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
                segment_ids.append(0)
                p_mask.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                    p_mask.append(1)
                tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
                segment_ids.append(0)
                p_mask.append(1)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i

                    cur_tok_start_to_orig_index.append(
                        tok_start_to_orig_index[split_token_index])
                    cur_tok_end_to_orig_index.append(
                        tok_end_to_orig_index[split_token_index])

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                                split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                    p_mask.append(0)
                tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
                segment_ids.append(1)
                p_mask.append(1)

                paragraph_len = len(tokens)
                input_ids = tokens

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    p_mask.append(1)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                span_is_impossible = example.is_impossible
                start_position = None
                end_position = None
                if is_training and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        # continue
                        start_position = 0
                        end_position = 0
                        span_is_impossible = True
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if is_training and span_is_impossible:
                    start_position = 0
                    end_position = 0

                if example_index < 20:
                    print("*** Example ***")
                    print("unique_id: %s" % (unique_id))
                    print("example_index: %s" % (example_index))
                    print("doc_span_index: %s" % (doc_span_index))
                    print("tok_start_to_orig_index: %s" % " ".join(
                        [str(x) for x in cur_tok_start_to_orig_index]))
                    print("tok_end_to_orig_index: %s" % " ".join(
                        [str(x) for x in cur_tok_end_to_orig_index]))
                    print("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    print("input_pieces: %s" % " ".join(
                        [tokenizer.sp_model.IdToPiece(x) for x in tokens]))
                    print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    print(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    print(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                    if is_training and span_is_impossible:
                        print("impossible example span")

                    if is_training and not span_is_impossible:
                        pieces = [tokenizer.sp_model.IdToPiece(token) for token in
                                  tokens[start_position: (end_position + 1)]]
                        answer_text = tokenizer.sp_model.DecodePieces(pieces)
                        print("start_position: %d" % (start_position))
                        print("end_position: %d" % (end_position))
                        print(
                            "answer: %s" % (tokenization.printable_text(answer_text)))

                        # note(zhiliny): With multi processing,
                        # the example_index is actually the index within the current process
                        # therefore we use example_index=None to avoid being used in the future.
                        # The current code does not use example_index of training data.
                if is_training:
                    feat_example_index = None
                else:
                    feat_example_index = example_index

                ep["unique_id"] = np.array([unique_id])
                ep["input_ids"] = np.array(input_ids)
                ep["input_mask"] = np.array(input_mask)
                ep["segment_ids"] = np.array(segment_ids)
                ep["p_mask"] = np.array(p_mask)
                if is_training:
                    ep["start_position"] = np.array([start_position])
                    ep["end_position"] = np.array([end_position])
                    if span_is_impossible:
                        span_is_impossible = 1
                    else:
                        span_is_impossible = 0
                    ep["is_impossible"] = np.array([span_is_impossible])
                data_input.append(ep)
                feature = InputFeaturesSquad(
                    unique_id=unique_id,
                    example_index=feat_example_index,
                    doc_span_index=doc_span_index,
                    tok_start_to_orig_index=cur_tok_start_to_orig_index,
                    tok_end_to_orig_index=cur_tok_end_to_orig_index,
                    token_is_max_context=token_is_max_context,
                    tokens=tokenizer.convert_ids_to_tokens(tokens),
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    p_mask=p_mask)

                # Run callback
                output.append(feature)
                unique_id += 1
                if span_is_impossible:
                    cnt_neg += 1
                else:
                    cnt_pos += 1
        if data_input != []:
            writer.write_raw_data(data_input)
        writer.commit()
        print("Total number of instances: {} = pos {} neg {}".format(
            cnt_pos + cnt_neg, cnt_pos, cnt_neg))
        return output


class RaceExample:
    """A single training/test example for the RACE dataset."""

    def __init__(self,
                 example_id,
                 context_sentence,
                 start_ending,
                 endings,
                 label=None):
        self.example_id = example_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = endings
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.example_id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


class InputFeaturesRace:
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 guid=None,
                 example_id=None,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_id = example_id
        self.guid = guid
        self.is_real_example = is_real_example


class PaddingInputExample:
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class RaceProcessor:
    """Processor for the RACE data set."""

    def __init__(self, use_spm, do_lower_case, high_only, middle_only):
        super(RaceProcessor, self).__init__()
        self.use_spm = use_spm
        self.do_lower_case = do_lower_case
        self.high_only = high_only
        self.middle_only = middle_only

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_examples(
            os.path.join(data_dir, "RACE", "train"))

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_examples(
            os.path.join(data_dir, "RACE", "dev"))

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        return self.read_examples(
            os.path.join(data_dir, "RACE", "test"))

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["A", "B", "C", "D"]

    def process_text(self, text):
        if self.use_spm:
            return tokenization.preprocess_text(text, lower=self.do_lower_case)

        return tokenization.convert_to_unicode(text)

    def read_examples(self, data_dir):
        """Read examples from RACE json files."""
        examples = []
        for level in ["middle", "high"]:
            if level == "middle" and self.high_only: continue
            if level == "high" and self.middle_only: continue
            cur_dir = os.path.join(data_dir, level)

            file_list = os.listdir(cur_dir)
            for file_name in file_list:
                if not file_name.endswith('.txt'):
                    continue
                cur_path = os.path.join(cur_dir, file_name)
                with open(cur_path) as f:
                    for line in f:
                        cur_data = json.loads(line.strip())

                        answers = cur_data["answers"]
                        options = cur_data["options"]
                        questions = cur_data["questions"]
                        context = self.process_text(cur_data["article"])

                        for i in range(len(answers)):
                            label = ord(answers[i]) - ord("A")
                            qa_list = []

                            question = self.process_text(questions[i])
                            for j in range(4):
                                option = self.process_text(options[i][j])

                                if "_" in question:
                                    qa_cat = question.replace("_", option)
                                else:
                                    qa_cat = " ".join([question, option])

                                qa_list.append(qa_cat)

                            examples.append(
                                RaceExample(
                                    example_id=cur_data["id"],
                                    context_sentence=context,
                                    start_ending=None,
                                    endings=[qa_list[0], qa_list[1], qa_list[2], qa_list[3]],
                                    label=label
                                )
                            )

        return examples

    def convert_race_examples_to_features(self,
                                          examples,
                                          label_list,
                                          tokenizer,
                                          max_seq_length,
                                          max_qa_length,
                                          output_file):
        """convert race examples to features"""
        writer = FileWriter(file_name=output_file, shard_num=1)
        data_schema = {"input_ids": {"type": "int64", "shape": [4, max_seq_length]},
                       "input_mask": {"type": "int64", "shape": [4, max_seq_length]},
                       "segment_ids": {"type": "int64", "shape": [4, max_seq_length]},
                       "label_id": {"type": "int64", "shape": [-1]},
                       "is_real_example": {"type": "int64", "shape": [-1]}}
        writer.add_schema(data_schema)
        data_input = []
        for (ex_index, example) in enumerate(examples):
            ep = {}
            if ex_index % 10000 == 0:
                print("Writing example %d of %d" % (ex_index, len(examples)))

            if isinstance(example, PaddingInputExample):
                label_size = len(label_list)
                all_input_ids = np.array([[0] * max_seq_length] * label_size)
                all_input_mask = np.array([[0] * max_seq_length] * label_size)
                all_segment_ids = np.array([[0] * max_seq_length] * label_size)
                label = np.array([0])
                is_real_example = np.array([0])
            else:
                context_tokens = tokenizer.tokenize(example.context_sentence)
                if example.start_ending is not None:
                    start_ending_tokens = tokenizer.tokenize(example.start_ending)

                all_input_tokens = []
                all_input_ids = []
                all_input_mask = []
                all_segment_ids = []
                for ending in example.endings:
                    # We create a copy of the context tokens in order to be
                    # able to shrink it according to ending_tokens
                    context_tokens_choice = context_tokens[:]
                    if example.start_ending is not None:
                        ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
                    else:
                        ending_tokens = tokenizer.tokenize(ending)
                    # Modifies `context_tokens_choice` and `ending_tokens` in
                    # place so that the total length is less than the
                    # specified length.  Account for [CLS], [SEP], [SEP] with
                    # "- 3"
                    ending_tokens = ending_tokens[- max_qa_length:]

                    if len(context_tokens_choice) + len(ending_tokens) > max_seq_length - 3:
                        context_tokens_choice = context_tokens_choice[: (max_seq_length - 3 - len(ending_tokens))]

                    tokens = ["[CLS]"] + context_tokens_choice + (["[SEP]"] + ending_tokens + ["[SEP]"])

                    segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    padding = [0] * (max_seq_length - len(input_ids))
                    input_ids += padding
                    input_mask += padding
                    segment_ids += padding

                    assert len(input_ids) == max_seq_length
                    assert len(input_mask) == max_seq_length
                    assert len(segment_ids) == max_seq_length

                    all_input_tokens.append(tokens)
                    all_input_ids.append(input_ids)
                    all_input_mask.append(input_mask)
                    all_segment_ids.append(segment_ids)

                label = example.label
                if ex_index < 5:
                    print("*** Example ***")
                    print("id: {}".format(example.example_id))
                    for choice_idx, (tokens, input_ids, input_mask, segment_ids) in \
                            enumerate(zip(all_input_tokens, all_input_ids, all_input_mask, all_segment_ids)):
                        print("choice: {}".format(choice_idx))
                        print("tokens: {}".format(" ".join(tokens)))
                        print(
                            "input_ids: {}".format(" ".join(map(str, input_ids))))
                        print(
                            "input_mask: {}".format(" ".join(map(str, input_mask))))
                        print(
                            "segment_ids: {}".format(" ".join(map(str, segment_ids))))
                        print("label: {}".format(label))

                is_real_example = 1

            ep["input_ids"] = np.array(all_input_ids)
            ep["input_mask"] = np.array(all_input_mask)
            ep["segment_ids"] = np.array(all_segment_ids)
            ep["label_id"] = np.array([label])
            ep["is_real_example"] = np.array([is_real_example])
            data_input.append(ep)
        writer.write_raw_data(data_input)
        writer.commit()





class ClassifierExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeaturesClassifier():
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 guid=None,
                 example_id=None,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_id = example_id
        self.guid = guid
        self.is_real_example = is_real_example


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, use_spm, do_lower_case):
        super(DataProcessor, self).__init__()
        self.use_spm = use_spm
        self.do_lower_case = do_lower_case

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def process_text(self, text):
        if self.use_spm:
            return tokenization.preprocess_text(text, lower=self.do_lower_case)

        return tokenization.convert_to_unicode(text)


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MNLI", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MNLI", "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MNLI", "test_matched.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # Note(mingdachen): We will rely on this guid for GLUE submission.
            guid = self.process_text(line[0])
            text_a = self.process_text(line[8])
            text_b = self.process_text(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = self.process_text(line[-1])
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MisMnliProcessor(MnliProcessor):
    """Processor for the Mismatched MultiNLI data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MNLI", "dev_mismatched.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MNLI", "test_mismatched.tsv")),
            "test")


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MRPC", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MRPC", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MRPC", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = self.process_text(line[3])
            text_b = self.process_text(line[4])
            if set_type == "test":
                guid = line[0]
                label = "0"
            else:
                label = self.process_text(line[0])
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CoLA", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CoLA", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CoLA", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                guid = line[0]
                text_a = self.process_text(line[1])
                label = "0"
            else:
                text_a = self.process_text(line[3])
                label = self.process_text(line[1])
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "SST-2", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "SST-2", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "SST-2", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if set_type != "test":
                guid = "%s-%s" % (set_type, i)
                text_a = self.process_text(line[0])
                label = self.process_text(line[1])
            else:
                guid = self.process_text(line[0])
                # guid = "%s-%s" % (set_type, line[0])
                text_a = self.process_text(line[1])
                label = "0"
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "STS-B", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "STS-B", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "STS-B", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = self.process_text(line[0])
            # guid = "%s-%s" % (set_type, line[0])
            text_a = self.process_text(line[7])
            text_b = self.process_text(line[8])
            if set_type != "test":
                label = float(line[-1])
            else:
                label = 0
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "QQP", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "QQP", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "QQP", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            # guid = "%s-%s" % (set_type, line[0])
            if set_type != "test":
                try:
                    text_a = self.process_text(line[3])
                    text_b = self.process_text(line[4])
                    label = self.process_text(line[5])
                except IndexError:
                    continue
            else:
                text_a = self.process_text(line[1])
                text_b = self.process_text(line[2])
                label = "0"
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "QNLI", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "QNLI", "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "QNLI", "test.tsv")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = self.process_text(line[0])
            # guid = "%s-%s" % (set_type, line[0])
            text_a = self.process_text(line[1])
            text_b = self.process_text(line[2])
            if set_type == "test_matched":
                label = "entailment"
            else:
                label = self.process_text(line[-1])
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "RTE", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "RTE", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "RTE", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = self.process_text(line[0])
            # guid = "%s-%s" % (set_type, line[0])
            text_a = self.process_text(line[1])
            text_b = self.process_text(line[2])
            if set_type == "test":
                label = "entailment"
            else:
                label = self.process_text(line[-1])
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WNLI", "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WNLI", "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WNLI", "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = self.process_text(line[0])
            # guid = "%s-%s" % (set_type, line[0])
            text_a = self.process_text(line[1])
            text_b = self.process_text(line[2])
            if set_type != "test":
                label = self.process_text(line[-1])
            else:
                label = "0"
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AXProcessor(DataProcessor):
    """Processor for the AX data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        pass

    def get_train_examples(self, data_dir):
        pass

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "diagnostic", "diagnostic.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # Note(mingdachen): We will rely on this guid for GLUE submission.
            guid = self.process_text(line[0])
            text_a = self.process_text(line[1])
            text_b = self.process_text(line[2])
            if set_type == "test":
                label = "contradiction"
            else:
                label = self.process_text(line[-1])
            examples.append(
                ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_classifier_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, vocab_file, task_name, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = FileWriter(file_name=output_file, shard_num=1)
    if task_name.lower() == "sts-b":
        label_schema = {"type": "int64", "shape": [-1]}
    else:
        label_schema = {"type": "float32", "shape": [-1]}
    data_schema = {"input_ids": {"type": "int64", "shape": [max_seq_length]},
                   "input_mask": {"type": "int64", "shape": [max_seq_length]},
                   "segment_ids": {"type": "int64", "shape": [max_seq_length]},
                   "label_id": label_schema,
                   "is_real_example": {"type": "int64", "shape": [-1]}}
    writer.add_schema(data_schema)
    data_input = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        ep = {}
        if isinstance(example, PaddingInputExample):
            label_size = len(label_list)
            input_ids = np.array([[0] * max_seq_length] * label_size)
            input_mask = np.array([[0] * max_seq_length] * label_size)
            segment_ids = np.array([[0] * max_seq_length] * label_size)
            label_id = np.array([0])
            is_real_example = np.array([0])
        else:
            if task_name.lower() != "sts-b":
                label_map = {}
                for (i, label) in enumerate(label_list):
                    label_map[label] = i

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in ALBERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if task_name.lower() != "sts-b":
                label_id = label_map[example.label]
            else:
                label_id = example.label
            is_real_example = np.array([1])
            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (example.guid))
                print("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("label: %s (id = %d)" % (example.label, label_id))

            ep["input_ids"] = np.array(input_ids)
            ep["input_mask"] = np.array(input_mask)
            ep["segment_ids"] = np.array(segment_ids)
            ep["label_id"] = np.array([label_id])
            ep["is_real_example"] = np.array([is_real_example])
            data_input.append(ep)
            # writer.write_raw_data(data_input)
    writer.write_raw_data(data_input)
    writer.commit()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
