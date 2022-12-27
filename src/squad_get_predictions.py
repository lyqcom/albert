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

"""get predictions for squad"""

import math
import collections
import os
import six
from src import tokenization


def get_prelim_predictions(features, unique_id_to_result, n_best_size, max_answer_length):
    """get prelim predictions"""
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    for (feature_index, feature) in enumerate(features):
        if feature.unique_id not in unique_id_to_result:
            continue
        result = unique_id_to_result[feature.unique_id]
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)
    return prelim_predictions


def get_nbest(prelim_predictions, features, example, n_best_size, do_lower_case):
    """get nbest predictions"""
    _NbestPrediction = collections.namedtuple(
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1
    return nbest


def accumulate_predictions_v1(result_dict, all_examples, all_features,
                              all_results, n_best_size, max_answer_length):
    """accumulate predictions for each positions in a dictionary."""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    for example_index, features in example_index_to_features.items():
        if example_index not in result_dict:
            result_dict[example_index] = {}
        # features = example_index_to_features[example_index]

        for (_, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            # if feature.unique_id not in unique_id_to_result.keys():
            #     continue
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


def write_predictions_v1(result_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         output_prediction_file, output_nbest_file):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for example_index, features in example_index_to_features.items():
        example = all_examples[example_index]

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

    # with open(output_prediction_file, "w") as writer:
    #   writer.write(json.dumps(all_predictions, indent=4) + "\n")
    #
    # with open(output_nbest_file, "w") as writer:
    #   writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_dir):
    """Write final predictions to the json file and log-odds of null if needed."""
    output_prediction_file = os.path.join(
        output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        output_dir, "nbest_predictions.json")
    result_dict = {}
    accumulate_predictions_v1(
        result_dict, all_examples, all_features,
        all_results, n_best_size, max_answer_length)
    all_predictions = write_predictions_v1(
        result_dict, all_examples, all_features, all_results,
        n_best_size, max_answer_length,
        output_prediction_file, output_nbest_file)
    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for (i, score) in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(score[0])
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
