# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
"""Utility functions for SQuAD v1.1/v2.0 datasets."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import collections
import json
import math
import re
import string
import sys
import six
from six.moves import range

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "start_log_prob",
                                    "end_log_prob"])

RawResultV2 = collections.namedtuple(
    "RawResultV2",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


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


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def accumulate_predictions_v1(result_dict, all_examples, all_features,
                              all_results, n_best_size, max_answer_length):
    """accumulate predictions for each positions in a dictionary."""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    for (example_index, _) in enumerate(all_examples):
        if example_index not in result_dict:
            result_dict[example_index] = {}
        features = example_index_to_features[example_index]

        for (_, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_log_prob, n_best_size)
            end_indexes = _get_best_indexes(result.end_log_prob, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    doc_offset = feature.tokens.index("[SEP]") + 1
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
                        continue
                    if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_log_prob = result.start_log_prob[start_index]
                    end_log_prob = result.end_log_prob[end_index]
                    start_idx = start_index - doc_offset
                    end_idx = end_index - doc_offset
                    if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
                        result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
                    result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append(
                        (start_log_prob, end_log_prob))


def write_predictions_v1(result_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         output_prediction_file, output_nbest_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        for (feature_index, feature) in enumerate(features):
            for ((start_idx, end_idx), logprobs) in \
                    result_dict[example_index][feature.unique_id].items():
                start_log_prob = 0
                end_log_prob = 0
                for logprob in logprobs:
                    start_log_prob += logprob[0]
                    end_log_prob += logprob[1]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_idx,
                        end_index=end_idx,
                        start_log_prob=start_log_prob / len(logprobs),
                        end_log_prob=end_log_prob / len(logprobs)))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index >= 0:  # this is a non-null prediction
                tok_start_to_orig_index = feature.tok_start_to_orig_index
                tok_end_to_orig_index = feature.tok_end_to_orig_index
                start_orig_pos = tok_start_to_orig_index[pred.start_index]
                end_orig_pos = tok_end_to_orig_index[pred.end_index]

                paragraph_text = example.paragraph_text
                final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_log_prob=0.0, end_log_prob=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


####### following are from official SQuAD v1.1 evaluation scripts
def normalize_answer_v1(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """calculate f1 score"""
    prediction_tokens = normalize_answer_v1(prediction).split()
    ground_truth_tokens = normalize_answer_v1(ground_truth).split()
    common = (collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer_v1(prediction) == normalize_answer_v1(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_v1(dataset, predictions):
    """evaluation for squad v1.1"""
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = ("Unanswered question " + six.ensure_str(qa["id"]) +
                               "  will receive score 0.")
                    print(message, file=sys.stderr)
                    continue
                ground_truths = [x["text"] for x in qa["answers"]]
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score,
                                                             prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


####### above are from official SQuAD v1.1 evaluation scripts
####### following are from official SQuAD v2.0 evaluation scripts
def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer_v2(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer_v2(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer_v2(a_gold) == normalize_answer_v2(a_pred))


def compute_f1(a_gold, a_pred):
    """compute f1 score"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    """get_raw_scores"""
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer_v2(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    """make eval dict"""
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])

    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """find_best_thresh"""
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores: continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


####### above are from official SQuAD v2.0 evaluation scripts

def accumulate_predictions_v2(result_dict, cls_dict, all_examples,
                              all_features, all_results, n_best_size,
                              max_answer_length, start_n_top, end_n_top):
    """accumulate predictions for each positions in a dictionary."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    for (example_index, _) in enumerate(all_examples):
        if example_index not in result_dict:
            result_dict[example_index] = {}
        features = example_index_to_features[example_index]

        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (_, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            doc_offset = feature.tokens.index("[SEP]") + 1
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
                        continue
                    if start_index - doc_offset < 0:
                        continue
                    if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_idx = start_index - doc_offset
                    end_idx = end_index - doc_offset
                    if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
                        result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
                    result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append(
                        (start_log_prob, end_log_prob))
        if example_index not in cls_dict:
            cls_dict[example_index] = []
        cls_dict[example_index].append(score_null)


def write_predictions_v2(result_dict, cls_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         output_prediction_file,
                         output_nbest_file, output_null_log_odds_file,
                         null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        # score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            for ((start_idx, end_idx), logprobs) in \
                    result_dict[example_index][feature.unique_id].items():
                start_log_prob = 0
                end_log_prob = 0
                for logprob in logprobs:
                    start_log_prob += logprob[0]
                    end_log_prob += logprob[1]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_idx,
                        end_index=end_idx,
                        start_log_prob=start_log_prob / len(logprobs),
                        end_log_prob=end_log_prob / len(logprobs)))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_start_to_orig_index = feature.tok_start_to_orig_index
            tok_end_to_orig_index = feature.tok_end_to_orig_index
            start_orig_pos = tok_start_to_orig_index[pred.start_index]
            end_orig_pos = tok_end_to_orig_index[pred.end_index]

            paragraph_text = example.paragraph_text
            final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="",
                    start_log_prob=-1e6,
                    end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = sum(cls_dict[example_index]) / len(cls_dict[example_index])
        scores_diff_json[example.qas_id] = score_diff
        # predict null answers when null threshold is provided
        if null_score_diff_threshold is None or score_diff < null_score_diff_threshold:
            all_predictions[example.qas_id] = best_non_null_entry.text
        else:
            all_predictions[example.qas_id] = ""

        all_nbest_json[example.qas_id] = nbest_json
        assert len(nbest_json) >= 1

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    with open(output_null_log_odds_file, "w") as writer:
        writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    return all_predictions, scores_diff_json


def evaluate_v2(result_dict, cls_dict, prediction_json, eval_examples,
                eval_features, all_results, n_best_size, max_answer_length,
                output_prediction_file, output_nbest_file,
                output_null_log_odds_file):
    """evaluation for squad v2.0"""
    null_score_diff_threshold = None
    predictions, na_probs = write_predictions_v2(
        result_dict, cls_dict, eval_examples, eval_features,
        all_results, n_best_size, max_answer_length,
        output_prediction_file, output_nbest_file,
        output_null_log_odds_file, null_score_diff_threshold)

    na_prob_thresh = 1.0  # default value taken from the eval script
    qid_to_has_ans = make_qid_to_has_ans(prediction_json)  # maps qid to True/False
    exact_raw, f1_raw = get_raw_scores(prediction_json, predictions)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, na_probs, qid_to_has_ans)
    null_score_diff_threshold = out_eval["best_f1_thresh"]

    predictions, na_probs = write_predictions_v2(
        result_dict, cls_dict, eval_examples, eval_features,
        all_results, n_best_size, max_answer_length,
        output_prediction_file, output_nbest_file,
        output_null_log_odds_file, null_score_diff_threshold)

    qid_to_has_ans = make_qid_to_has_ans(prediction_json)  # maps qid to True/False
    exact_raw, f1_raw = get_raw_scores(prediction_json, predictions)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    out_eval["null_score_diff_threshold"] = null_score_diff_threshold
    return out_eval
