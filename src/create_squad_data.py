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

"""create squad data"""

import collections
import json
import six

import numpy as np

import tokenization


class SquadExample:
    """extract column contents from raw data"""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class InputFeatures:
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


def read_squad_examples(input_file, is_training, version_2_with_negative=False):
    """Return list of SquadExample from input_data or input_file (SQuAD json file)"""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def token_offset(text):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return (doc_tokens, char_to_word_offset)

    def process_one_example(qa, is_training, version_2_with_negative, doc_tokens, char_to_word_offset):
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = -1
        end_position = -1
        orig_answer_text = ""
        is_impossible = False
        if is_training:
            if version_2_with_negative:
                is_impossible = qa["is_impossible"]
            if (len(qa["answers"]) != 1) and (not is_impossible):
                raise ValueError("For training, each question should have exactly 1 answer.")
            if not is_impossible:
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    return None
        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        return example

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens, char_to_word_offset = token_offset(paragraph_text)
            for qa in paragraph["qas"]:
                one_example = process_one_example(qa, is_training, version_2_with_negative,
                                                  doc_tokens, char_to_word_offset)
                if one_example is not None:
                    examples.append(one_example)
    return examples


# def _check_is_max_context(doc_spans, cur_span_index, position):
#     """Check if this is the 'max context' doc span for the token."""
#     best_score = None
#     best_span_index = None
#     for (span_index, doc_span) in enumerate(doc_spans):
#         end = doc_span.start + doc_span.length - 1
#         if position < doc_span.start:
#             continue
#         if position > end:
#             continue
#         num_left_context = position - doc_span.start
#         num_right_context = end - position
#         score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
#         if best_score is None or score > best_score:
#             best_score = score
#             best_span_index = span_index
#     return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _convert_index(index, pos, m=None, is_start=True):
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


def _check_is_max_context(doc_spans, cur_span_index, position):
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


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride,
                                 max_query_length, is_training, output_fn, do_lower_case):
    """Loads a data file into a list of `InputBatch`s."""
    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_n, max_m = 1024, 1024
    f = np.zeros((max_n, max_m), dtype=np.float32)
    output = []
    g = {}

    def _lcs_match(max_dist, n, m):
        """Longest-common-substring algorithm."""
        f.fill(0)
        g.clear()

        ### longest common sub sequence
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
        query_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
            tokenization.preprocess_text(
                example.question_text, lower=do_lower_case)))

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        paragraph_text = example.paragraph_text
        para_tokens = tokenizer.tokenize(
            tokenization.preprocess_text(
                example.paragraph_text, lower=do_lower_case))

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

        tok_cat_text = " ".join(para_tokens)
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
            start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                            n, is_start=True)
            end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
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

            start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                               is_start=True)
            tok_start_position = chartok_to_tok_index[start_chartok_pos]

            end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                             is_start=False)
            tok_end_position = chartok_to_tok_index[end_chartok_pos]
            assert tok_start_position <= tok_end_position

        def _piece_to_id(x):
            new_x = []
            for _, item in enumerate(x):
                if six.PY2 and isinstance(item, six.text_type):
                    new_x.append(six.ensure_binary(item, "utf-8"))
                else:
                    new_x.append(item)
            return tokenizer.convert_tokens_to_ids(new_x)

        all_doc_tokens = _piece_to_id(para_tokens)

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
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []

            cur_tok_start_to_orig_index = []
            cur_tok_end_to_orig_index = []

            tokens.extend(tokenizer.convert_tokens_to_ids(["[CLS]"]))
            segment_ids.append(0)
            p_mask.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                p_mask.append(1)
            tokens.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
            segment_ids.append(0)
            p_mask.append(1)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                cur_tok_start_to_orig_index.append(
                    tok_start_to_orig_index[split_token_index])
                cur_tok_end_to_orig_index.append(
                    tok_end_to_orig_index[split_token_index])

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                p_mask.append(0)
            tokens.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
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
            if is_training:
                feat_example_index = None
            else:
                feat_example_index = example_index
            feature = InputFeatures(
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
    return output
