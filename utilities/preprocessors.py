""" Data processors and helpers """
import csv
import json
import logging
import os

import html
import sys

from tqdm import tqdm
from transformers import glue_processors, glue_output_modes
from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from sklearn.model_selection import train_test_split
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        # processor = glue_processors[task]()
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,
                                       truncation = True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                try:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[0]
                    X_test.append([text_a, text_b])
                    y_test.append(label)
                except IndexError:
                    continue
                X_test.append([text_a, text_b])
                y_test.append(label)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[0]
                X.append([text_a, text_b])
                Y.append(label)
            except IndexError:
                continue

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if type(line[0][0]) is not str:
                text_a = line[0][0][0]
                text_b = line[0][1][0]
                label = line[1]
            else:
                try:
                    text_a = line[0][0]
                    text_b = line[0][1]
                    label = line[1]
                except IndexError:
                    continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# class MnliProcessor(DataProcessor):
#     """Processor for the MultiNLI data set (GLUE version)."""

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir, ood=False):
        filename_json = 'test' if not ood else 'test_ood'
        filename_tsv = 'dev_matched' if not ood else 'dev_mismatched'
        if os.path.isfile(os.path.join(data_dir, "{}.json".format(filename_json))):
            with open(os.path.join(data_dir, "{}.json".format(filename_json))) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "{}.tsv".format(filename_tsv)))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                X_test.append([line[8], line[9]])
                y_test.append(line[-1])

            with open(os.path.join(data_dir, "{}.json".format(filename_json)), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_test_examples_ood(self, data_dir, ood=True):
        filename_json = 'test' if not ood else 'test_ood'
        filename_tsv = 'dev_matched' if not ood else 'dev_mismatched'
        if os.path.isfile(os.path.join(data_dir, "{}.json".format(filename_json))):
            with open(os.path.join(data_dir, "{}.json".format(filename_json))) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "{}.tsv".format(filename_tsv)))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                X_test.append([line[8], line[9]])
                y_test.append(line[-1])

            with open(os.path.join(data_dir, "{}.json".format(filename_json)), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def _train_dev_split(self, data_dir, seed=42, split=0.05):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            X.append([line[8], line[9]])
            Y.append(line[-1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train, Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val, Y_val], f)

        return

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, line[0])
            if type(line[0][0]) is not str:
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
            else:
                text_a = line[0][0]
                text_b = line[0][1]
                label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    # def get_example_from_tensor_dict(self, tensor_dict):
    #     """See base class."""
    #     return InputExample(
    #         tensor_dict["idx"].numpy(),
    #         tensor_dict["premise"].numpy().decode("utf-8"),
    #         tensor_dict["hypothesis"].numpy().decode("utf-8"),
    #         str(tensor_dict["label"].numpy()),
    #     )
    #
    # def get_train_examples(self, data_dir):
    #     """See base class."""
    #     if not os.path.isfile(os.path.join(data_dir, "train_telephone_42.json")):
    #         self._train_dev_split(data_dir)
    #     with open(os.path.join(data_dir, "train_telephone_42.json")) as json_file:
    #         [X_train, y_train] = json.load(json_file)
    #     train_examples = self._create_examples(zip(X_train, y_train), "train")
    #     return train_examples
    #     # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    #
    # def _train_dev_split(self, data_dir, seed=42, split=0.05):
    #     """Splits train set into train and dev sets."""
    #     lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
    #     genres = set([l[3] for l in lines if l[3]!='genre'])
    #     X = []
    #     Y = []
    #     for (i, line) in enumerate(lines):
    #         if i == 0:
    #             continue
    #         if line[3] == 'telephone':
    #             X.append([line[8],line[9]])
    #             Y.append(line[-1])
    #
    #     X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)
    #
    #     # Write the train set into a json file (for this seed)
    #     with open(os.path.join(data_dir, "train_telephone_{}.json".format(seed)), "w") as f:
    #         json.dump([X_train , Y_train], f)
    #
    #     # Write the dev set into a json file (for this seed)
    #     with open(os.path.join(data_dir, "dev_telephone_{}.json".format(seed)), "w") as f:
    #         json.dump([X_val , Y_val], f)
    #
    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     if not os.path.isfile(os.path.join(data_dir, "dev_telephone_42.json")):
    #         self._train_dev_split(data_dir)
    #     with open(os.path.join(data_dir, "dev_telephone_42.json")) as json_file:
    #         [X_val, y_val] = json.load(json_file)
    #     dev_examples = self._create_examples(zip(X_val, y_val), "dev")
    #     return dev_examples
    #     # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
    #
    # def get_test_examples(self, data_dir):
    #     if os.path.isfile(os.path.join(data_dir, "test_telephone.json")):
    #         with open(os.path.join(data_dir, "test_telephone.json")) as json_file:
    #             [X_test, y_test] = json.load(json_file)
    #     else:
    #
    #         lines = self._read_tsv(os.path.join(data_dir, "dev_matched.tsv"))
    #         X_test = []
    #         y_test = []
    #         for (i, line) in enumerate(lines):
    #             if i == 0:
    #                 continue
    #             if line[3] == 'telephone':
    #                 X_test.append([line[8], line[9]])
    #                 y_test.append(line[-1])
    #         with open(os.path.join(data_dir, "test_telephone.json"), "w") as f:
    #             json.dump([X_test, y_test], f)
    #
    #     test_examples = self._create_examples(zip(X_test, y_test), "test")
    #     return test_examples
    #
    # def get_test_examples_ood(self, data_dir):
    #     if os.path.isfile(os.path.join(data_dir, "test_letters_f2f.json")):
    #         with open(os.path.join(data_dir, "test_letters_f2f.json")) as json_file:
    #             [X_test, y_test] = json.load(json_file)
    #     else:
    #
    #         lines = self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv"))
    #         X_test = []
    #         y_test = []
    #         for (i, line) in enumerate(lines):
    #             if i == 0:
    #                 continue
    #             if line[3] in ['letters', 'facetoface']:
    #                 X_test.append([line[8], line[9]])
    #                 y_test.append(line[-1])
    #
    #         with open(os.path.join(data_dir, "test_letters_f2f.json"), "w") as f:
    #             json.dump([X_test, y_test], f)
    #     test_examples = self._create_examples(zip(X_test, y_test), "test")
    #     return test_examples
    #
    # def get_augm_examples(self, X, y):
    #     return self._create_examples(zip(X,y), "augm")
    #
    # def get_labels(self):
    #     """See base class."""
    #     return ["contradiction", "entailment", "neutral"]
    #
    # def _create_examples(self, lines, set_type):
    #     """Creates examples for the training and dev sets."""
    #     examples = []
    #     for (i, line) in enumerate(lines):
    #         # if i == 0:
    #         #     continue
    #         guid = "%s-%s" % (set_type, line[0])
    #         # text_a = line[8]
    #         # text_b = line[9]
    #         # label = line[-1]
    #         text_a = line[0][0]
    #         text_b = line[0][1]
    #         label = line[-1]
    #         examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    #     return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                X_test.append(line[0])
                y_test.append(line[1])

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line[0]) is not str:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
            else:
                if i == 0:
                    continue
                text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            X.append(line[0])
            Y.append(line[1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                try:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
                    X_test.append([text_a, text_b])
                    y_test.append(label)
                except IndexError:
                    continue
                X_test.append([text_a, text_b])
                y_test.append(label)

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
                X.append([text_a, text_b])
                Y.append(label)
            except IndexError:
                continue

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, line[0])
            if type(line[0][0]) is not str:
                text_a = line[0][0][0]
                text_b = line[0][1][0]
                label = line[1]
            else:
                try:
                    text_a = line[0][0]
                    text_b = line[0][1]
                    label = line[1]
                except IndexError:
                    continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    # def get_train_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                X_test.append([line[1],line[2]])
                y_test.append(line[-1])

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def _train_dev_split(self, data_dir, seed=42, split=0.05):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            X.append([line[1],line[2]])
            Y.append(line[-1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if type(line[0][0]) is not str:
                text_a = line[0][0][0]
                text_b = line[0][1][0]
            else:
                text_a = line[0][0]
                text_b = line[0][1]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        return self._create_examples(zip(X_train, y_train), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                X_test.append([line[1],line[2]])
                y_test.append(line[-1])

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def _train_dev_split(self, data_dir, seed=42, split=0.05):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            X.append([line[1],line[2]])
            Y.append(line[-1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type,i)
            # text_a = line[1]
            # text_b = line[2]
            text_a = line[0][0]
            text_b = line[0][1]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks = ["cola", "mnli", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli"]


class Trec6Processor(DataProcessor):
    """Processor for the TREC-6 data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def read_txt(cls, input_file):
        """Reads a text file."""
        # return open(input_file, "r", encoding="utf-8").readlines()
        return open(input_file, "r", encoding="ISO-8859-1").readlines()

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.txt")):
            self._train_dev_split(data_dir)
        return self._create_examples(self.read_txt(os.path.join(data_dir, "train_42.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.txt")):
            self._train_dev_split(data_dir)
        return self._create_examples(self.read_txt(os.path.join(data_dir, "dev_42.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def get_labels(self):
        """See base class."""
        return ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line[0]) is list:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
                label = line[1]
            else:
                columns = line.rstrip().split(" ")
                text_a = " ".join(columns[2:])
                label = columns[0].split(":")[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        # lines = open(os.path.join(data_dir, "train.txt"), "r", encoding="utf-8").readlines()
        lines = open(os.path.join(data_dir, "train.txt"), "r", encoding="ISO-8859-1").readlines()
        for line in lines:
            columns = line.rstrip().split(" ")
            Y.append(columns[0].split(":")[0])
            X.append(" ".join(columns[1:]))
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a txt file (for this seed)
        with open(os.path.join(data_dir, "train_{}.txt".format(seed)), "a+") as writer:
            for item in zip(X_train, Y_train):
                writer.write("%s  %s\n" % (item[1], item[0]))

        # Write the dev set into a txt file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.txt".format(seed)), "a+") as writer:
            for item in zip(X_val, Y_val):
                writer.write("%s  %s\n" % (item[1], item[0]))
        return
        # # Create examples for the remaining train set (for this seed)
        # train_examples = []
        # for (i, sample) in enumerate(zip(X_train, Y_train)):
        #     guid = "%s-%s" % ("train", i)dev
        #     text_a, label = sample
        #     train_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        # return train_examples



class AgnewsProcessor(DataProcessor):
    """Processor for the AG NEWS data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)

        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            from nlp import load_dataset
            dataset = load_dataset('ag_news', split='test')
            X_test = [x['text'] for x in dataset]
            y_test = [x['label'] for x in dataset]

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def _train_dev_split(self, data_dir, seed=42, split=0.05):
        """Splits train set into train and dev sets."""
        from nlp import load_dataset
        dataset = load_dataset('ag_news', split='train')
        X = [x['text'] for x in dataset]
        Y = [x['label'] for x in dataset]

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return


    def get_labels(self):
        """See base class."""
        # return ["1", "2", "3", "4"]
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if type(line[0]) is not str:
                text_a = line[0][0]
            else:
                text_a = line[0]
            label = str(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class DBpediaProcessor(DataProcessor):
    """Processor for the PubMed data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        # X, Y = [], []
        # lines = self._read_csv(os.path.join(data_dir, "train.csv"))
        # for line in lines:
        #     X.append(",".join(line[1:]).rstrip())
        #     Y.append(line[0])
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)

        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:
            X_test, y_test = [], []
            lines = self._read_csv(os.path.join(data_dir, "test.csv"))
            for line in lines:
                X_test.append(",".join(line[1:]).rstrip())
                y_test.append(line[0])
            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")
    # def get_train_dev_examples(self, data_dir, seed, split=0.1, dom=-1):
    #     """Splits train set into train and dev sets."""
    #     X, Y = [], []
    #     lines = self._read_csv(os.path.join(data_dir, "train.csv"))
    #     for line in lines:
    #         X.append(",".join(line[1:]).rstrip())
    #         Y.append(line[0])
    #     X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)
    #     train_examples = self._create_examples(zip(X_train, Y_train), "train", dom)
    #     dev_examples = self._create_examples(zip(X_dev, Y_dev), "dev", dom)
    #     return train_examples, dev_examples
    #
    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line) is not str:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
                label = line[1]
            else:
                text_a, label = line
            # if dom != -1:
            #     label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42, split=0.05):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self._read_csv(os.path.join(data_dir, "train.csv"))
        for line in lines:
            X.append(",".join(line[1:]).rstrip())
            Y.append(line[0])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

class ImdbProcessor(DataProcessor):
    """Processor for the PubMed data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        # X, Y = [], []
        # lines = self._read_csv(os.path.join(data_dir, "train", "train.tsv"))
        # for line in lines[1:]:
        #     # X.append(",".join(line[1:]).rstrip())
        #     X.append(",".join(line[:-1]).rstrip())
        #     # Y.append(line[0])
        #     Y.append(line[-1])
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples

    # def get_contrast_examples(self, file=None, ori=False, data_dir=IMDB_CONTR_DATA_DIR):
    #     # if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
    #     #     self._train_dev_split(data_dir)
    #     # with open(os.path.join(data_dir, "dev_42.json")) as json_file:
    #     #     [X_val, y_val] = json.load(json_file)
    #     prefix='original' if ori else 'contrast'
    #     X, Y = [], []
    #     lines = self._read_csv(os.path.join(data_dir, "{}_{}.tsv".format(file, prefix)))
    #     labelname2int={"Positive":"1", "Negative":"0"}
    #     for i, line in enumerate(lines):
    #         if i == 0:
    #             continue
    #         # X.append(",".join(line[1:]).rstrip())
    #         X.append(",".join(line).rstrip().split('\t')[1])
    #         # Y.append(line[0])
    #         Y.append(labelname2int[",".join(line).rstrip().split('\t')[0]])
    #     dev_examples = self._create_examples(zip(X, Y), "{}_{}".format(file, prefix))
    #     return dev_examples

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            X_test, y_test = [], []
            lines = self._read_csv(os.path.join(data_dir, "test", "test.tsv"))
            for line in lines[1:]:
                # X.append(",".join(line[1:]).rstrip())
                X_test.append(",".join(line[:-1]).rstrip())
                # Y.append(line[0])
                y_test.append(line[-1])

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")

        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line) is not str:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
                label = line[1]
            else:
                text_a, label = line
                # if dom != -1:
                #     label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self._read_csv(os.path.join(data_dir, "train", "train.tsv"))
        for line in lines[1:]:
            # X.append(",".join(line[1:]).rstrip())
            X.append(",".join(line[:-1]).rstrip())
            # Y.append(line[0])
            Y.append(line[-1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

class TwitterPPDBProcessor(DataProcessor):
    """Processor for the PubMed data set."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        # X, Y = [], []
        # lines = self._read_csv(os.path.join(data_dir, "train", "train.tsv"))
        # for line in lines[1:]:
        #     # X.append(",".join(line[1:]).rstrip())
        #     X.append(",".join(line[:-1]).rstrip())
        #     # Y.append(line[0])
        #     Y.append(line[-1])
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            X_test, y_test = [], []
            samples = []
            path = os.path.join(data_dir, 'Twitter_URL_Corpus_test.txt')
            with open(path, newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                desc = f'loading \'{path}\''
                for row in tqdm(reader, desc=desc):
                    try:
                        sentence1 = row[0]
                        sentence2 = row[1]
                        label = eval(row[2])[0]
                        if self.valid_inputs(sentence1, sentence2, label):
                            label = 0 if label < 3 else 1
                            samples.append((sentence1, sentence2, label))
                            X_test.append([sentence1, sentence2])
                            y_test.append(str(label))
                    except:
                        pass
            # return samples

            # lines = self._read_csv(os.path.join(data_dir, "test", "test.tsv"))
            # for line in lines[1:]:
            #     # X.append(",".join(line[1:]).rstrip())
            #     X_test.append(",".join(line[:-1]).rstrip())
            #     # Y.append(line[0])
            #     y_test.append(line[-1])

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")

        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line) is not str:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
                label = line[1]
            else:
                text_a = line[0][0]
                text_b = line[0][1]
                label = line[1]
                # text_a, label = line
                # if dom != -1:
                #     label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        X, Y = [], []

        samples = []
        path = os.path.join(data_dir, 'Twitter_URL_Corpus_train.txt')
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        # return samples

        lines = self._read_csv(os.path.join(data_dir, "train", "train.tsv"))
        for line in lines[1:]:
            # X.append(",".join(line[1:]).rstrip())
            X.append(",".join(line[:-1]).rstrip())
            # Y.append(line[0])
            Y.append(line[-1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

class PubmedProcessor(DataProcessor):
    def get_labels(self):
        labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
        return labels
    def get_train_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")


class CounterfactualNliProcessor(DataProcessor):
    def ind2path(self, counterfactual):
        assert counterfactual in ["combined", "new", "orig"], 'counterfactual {}'.format(counterfactual)
        if counterfactual == 'orig':
            dir = "original"
        elif counterfactual == 'combined':
            dir = 'all_combined'
        elif counterfactual == 'new':
            dir = 'revised_combined'
        else:
            NotImplementedError
        return dir
    def get_labels(self):
        labels = ["entailment", "neutral", "contradiction"]
        return labels
    def get_train_examples(self, data_dir, counterfactual):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                 self.ind2path(counterfactual),
                                                                 "train.tsv")), "train")

    def get_dev_examples(self, data_dir, counterfactual):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                 self.ind2path(counterfactual),
                                                                 "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, counterfactual):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                 self.ind2path(counterfactual),
                                                                 "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i!=0:
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = line[1]
                label = line[2]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")


class CounterfactualSentimentProcessor(DataProcessor):
    def get_labels(self):
        labels = ["Negative", "Positive"]
        return labels
    def get_train_examples(self, data_dir, counterfactual):
        """See base class."""""
        assert counterfactual in ["combined", "new", "orig", "paired"], 'counterfactual {}'.format(counterfactual)
        if counterfactual == "paired":
            return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                     'combined/paired',
                                                                     "train_paired.tsv")), "train")
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                     counterfactual,
                                                                     "train.tsv")), "train")


    def get_dev_examples(self, data_dir, counterfactual):
        """See base class."""""
        if counterfactual == "paired":
            return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                     'combined/paired',
                                                                     "dev_paired.tsv")), "dev")
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                     counterfactual,
                                                                     "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, counterfactual):
        """See base class."""""
        if counterfactual == "paired":
            return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                     'combined/paired',
                                                                     "test_paired.tsv")), "test")
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir,
                                                                     counterfactual,
                                                                     "test.tsv")), "test")

    def get_test_examples_ood(self, data_dir, ood_dataset):
        """See base class."""""
        assert ood_dataset in ["amazon", "yelp", "semeval"]
        return self._create_examples(self._read_tsv(os.path.join(COUNTERFACTUAL_DATA_DIR,
                                                                 # counterfactual,
                                                                 '{}_balanced.tsv'.format(ood_dataset))), "test")
    def get_custom_examples(self, X, y):
        examples = []
        set_type="train"
        for (i, line) in enumerate(zip(X,y)):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            if label in self.get_labels():
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

        # return
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i!=0 and len(line)>=2:
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = line[0]
                if label in self.get_labels():
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    # "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "trec-6": Trec6Processor,
    "ag_news": AgnewsProcessor,
    "dbpedia": DBpediaProcessor,
    "imdb": ImdbProcessor,
    "twitterppdb": TwitterPPDBProcessor,
    "pubmed": PubmedProcessor,
    "nli": CounterfactualNliProcessor,
    "sentiment": CounterfactualSentimentProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "trec-6": "classification",
    "ag_news": "classification",
    "dbpedia": "classification",
    "imdb": "classification",
    "twitterppdb": "classification",
    "pubmed": "classification",
    "nli": "classification",
    "sentiment": "classification"
}
