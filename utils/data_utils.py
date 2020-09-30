import os
import logging

import torch
from torch.utils.data import TensorDataset


class InputExample(object):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, valid_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        # self.label_mask = label_mask


class NerProcessor:

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        '''
            MED : 医学
            IND : 工业类
            ECON : 经济类
            TECH : 技术类
            METE : 气象
        '''
        return ["0", "MED", "IND", "ECON", "TECH", "METE"]

    def _read_file(self, filename):
        '''
        read file
        '''
        f = open(filename, 'r', encoding='utf-8')
        data = []
        sentence = []
        label = []

        for i, line in enumerate(f.readlines()):

            splits = line.split("\1\t\1")
            if len(splits) < 2:
                print(line)
                break
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            title_cn, title_en, tag = splits[0], splits[1], splits[-1].strip()
            assert tag in self.get_labels(), "unknown tag {} in line {}".format(tag, i)

            sentence.append(title_cn.strip())
            label.append(tag.strip())

            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            sentence.append(title_en.strip())
            label.append(tag.strip())

            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []

        f.close()
        return data

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, encode_method):
    # ignored_label = "IGNORE"
    label_map = {label: i for i, label in enumerate(label_list)}
    # label_map[ignored_label] = 0  # 0 label is to be ignored

    features = []
    for (ex_index, example) in enumerate(examples):

        token_ids = encode_method(example.text_a)
        labels = [label_map[_label] for _label in example.label]
        valid = [1 for i in range(len(token_ids))]
        logging.debug("token ids = ")
        logging.debug(token_ids)
        logging.debug("labels = ")
        logging.debug(labels)
        logging.debug("valid = ")
        logging.debug(valid)

        if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
            token_ids = token_ids[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]

        # adding <s>
        token_ids.insert(0, 0)
        valid.insert(0, 0)

        # adding </s>
        token_ids.append(2)
        valid.append(0)

        assert len(labels) == 1

        input_mask = [1] * len(token_ids)

        while len(token_ids) < max_seq_length:
            token_ids.append(1)
            input_mask.append(0)
            valid.append(0)

        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(valid) == max_seq_length

        if ex_index < 2:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                [str(x) for x in token_ids]))
            logging.info("input_ids: %s" %
                         " ".join([str(x) for x in token_ids]))
            logging.info("input_mask: %s" %
                         " ".join([str(x) for x in input_mask]))
            logging.info("valid mask: %s" %
                         " ".join([str(x) for x in valid]))

        features.append(
            InputFeatures(input_ids=token_ids,
                          input_mask=input_mask,
                          label_id=labels,
                          valid_ids=valid))

    return features


def create_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids, all_valid_ids)
