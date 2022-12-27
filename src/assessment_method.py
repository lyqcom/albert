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
Albert evaluation assessment method script.
'''
import numpy as np
from mindspore.nn.metrics import Metric


class Accuracy(Metric):
    '''
    calculate accuracy
    '''

    def __init__(self):
        super(Accuracy, self).__init__()
        self.clear()

    def clear(self):
        self.total_num = 0
        self.acc_num = 0

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)

    def eval(self):
        return self.acc_num / self.total_num


class Spearman_Correlation():
    '''
    Calculate Spearman Correlation Coefficient
    '''

    def __init__(self):
        self.label = []
        self.logit = []

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.reshape(logits, -1)
        self.label.append(labels)
        self.logit.append(logits)

    def cal(self):
        '''
        Calculate Spearman Correlation
        '''
        label = np.concatenate(self.label)
        logit = np.concatenate(self.logit)
        sort_label = label.argsort()[::-1]
        sort_logit = logit.argsort()[::-1]
        n = len(label)
        d_acc = 0
        for i in range(n):
            d = np.where(sort_label == i)[0] - np.where(sort_logit == i)[0]
            d_acc += d ** 2
        ps = 1 - 6 * d_acc / n / (n ** 2 - 1)
        return ps


class Streaming_Pearson_Correlation(Metric):
    """Pearson Correlation"""
    def __init__(self):
        super(Streaming_Pearson_Correlation, self).__init__()
        self.clear()

    def clear(self):
        self.total_num = 0
        self.related_cofficient_sum = 0
        self.logits = np.array([])
        self.labels = np.array([])

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.reshape(logits, -1)
        self.labels = np.hstack((self.labels, labels))
        self.logits = np.hstack((self.logits, logits))

    def eval(self):
        return np.corrcoef(self.logits, self.labels)[0][1]


class CLSMetric(Metric):
    '''
    Self-defined Metric as a callback.
    '''

    def __init__(self):
        super(CLSMetric, self).__init__()
        self.clear()

    def clear(self):
        self.total_num = 0
        self.acc_num = 0

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)

    def eval(self):
        return self.acc_num / self.total_num
