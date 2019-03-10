# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
from pathlib import Path
import random
from io import open
import pickle
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from modeling import BertForMaskedLM
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear

import random

import tarfile
import requests


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




_CURPATH = Path.cwd() 
_TMPDIR = _CURPATH / "labada_data_intermediate"
_TRAINDIR = _TMPDIR
_TESTFILE = "lambada_development_plain_text.txt"
_DATADIR = _CURPATH / "labada_data"
_TAR = "lambada-dataset.tar.gz"
_URL = "http://clic.cimec.unitn.it/lambada/" + _TAR








class LambadaTest(Dataset):
    def __init__(self, corpus_path, testfile, tokenizer, seq_len, encoding="utf-8", corpus_lines=None,  rebuild=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0

        self.origlength = 0
        self.newlength = 0



        if rebuild:
            self.docs = []
            try:
                os.remove(self.corpus_path / "build_test_docs.p")
            except:
                pass
            
            with open(self.corpus_path / testfile, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.docs.append(line)

                    self.num_docs += 1

                    self.newlength += len(self.tokenizer.tokenize(line))
                    self.origlength += len(line.split())




            print(f"Testset prepared. Number of examples: {self.num_docs}")

            print(f"Length with Original Dataset Tokenization is: {self.origlength}. length with new tokenizsation: {self.newlength}")
        
            pickle.dump(self.docs, open(self.corpus_path / "build_test_docs.p", "wb"))
            print("Saved Dataset with Pickle")
        
        else:
            self.docs = pickle.load( open(self.corpus_path / "build_test_docs.p", "rb"))
            print("Loaded Dataset with Pickle")

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return len(self.docs)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1

        t1 = self.docs[item]

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)


        # transform sample to features
        cur_features = convert_example_to_features(tokens_a, self.sample_counter, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       )

        return cur_tensors






class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids



def convert_example_to_features(example, cur_time, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"




    tokens_a = truncate_seq_pair(tokens_a, max_seq_length - 3)




    savedtoken = []
    for l in range(5):
        if tokens_a[-1] in {".", ",", "'", "`" , "'", "?"}:
            savedtoken.insert(0, tokens_a[-1])
            tokens_a.pop()

        else:
            break


    lmlabel = tokens_a[-1]
    lmlabel = tokenizer.vocab[lmlabel]
    tokens_a.pop()


    # concatenate lm labels and account for CLS, SEP, SEP
    if not savedtoken:
        extra_lm_labels = 1
    else:
        extra_lm_labels = len(savedtoken)

    lm_label_ids = ([-1] + len(tokens_a)*[-1] + [lmlabel] + extra_lm_labels * [-1] + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)




    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[MASK]")
    if not savedtoken:
        tokens.append(".")
        segment_ids.append(0)
    else:
        tokens.extend(savedtoken)
        for _ in range(len(savedtoken)):
            segment_ids.append(0)
    tokens.append("[SEP]")

    
    segment_ids.append(0)
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    # print("input, segment, lmlabel")
    # print(len(input_ids))
    # print(len(segment_ids))
    # print(len(lm_label_ids))
    if len(input_ids) != max_seq_length:
        import pdb; pdb.set_trace()

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if cur_time < 5:
        logger.info("*** Example ***")
        logger.info("cur_time: %s" % (cur_time))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             )
    return features


def truncate_seq_pair(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        else:
            tokens_a = tokens_a[-max_length:].copy()
    return tokens_a





























def main():

    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--download",
                    action='store_true',
                    help="Whether to download the data again")
    parser.add_argument("--rebuild",
                    action='store_true',
                    help="whether to process the data again")
    parser.add_argument("--test",
                    action='store_true',
                    help="to activate test")

    args = parser.parse_args()




    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))



    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)




    num_train_optimization_steps = None

    train_dataset = LambadaTest(_TRAINDIR, _TESTFILE, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild)
    
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    # prepare model:

    model = BertForMaskedLM.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")


    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)



        
        model.eval()

        with torch.no_grad():

            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                wrong = 0
                right = 0
                totalcounteri = 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, lm_label_ids  = batch
                    loss, predictions = model(input_ids, segment_ids, input_mask, lm_label_ids)
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    print(f"Current Loss: {loss.item()}")

                    perpl = math.exp(loss.item())
                    print(f"Perplexity: {perpl}")
                    tr_loss += loss.item()

                    for res in range(predictions.size(0)):
                        #batch
                        ans = predictions[res]
                        #seperate all tokens
                        words = torch.chunk(ans, ans.size(0))
                        #return list of indixes of the best predicted word
                        words = [torch.argmax(x).item() for x in words]
                        # return list of best words
                        realwords = tokenizer.convert_ids_to_tokens(words)

                        
                        lindex = torch.argmax(lm_label_ids[res])
                        maskedword = lm_label_ids[res, lindex].item()
                        maskedword = tokenizer.ids_to_tokens[maskedword]

                        totalcounteri += 1

                        if maskedword[0] != "#":

                            accurate = maskedword == realwords[lindex]

                            if accurate:
                                right += 1
                            else:
                                wrong += 1

                    
                    predictions = predictions[0]
                    words = torch.chunk(predictions, predictions.size(0))
                    words = [torch.argmax(x).item() for x in words]
                    realwords = tokenizer.convert_ids_to_tokens(words)
                    print("Real sentences:")
                    
                    firstbatch = input_ids[0]
                    words = firstbatch.tolist()

                    actualwords = tokenizer.convert_ids_to_tokens(words)
                    joinedactualwords = " ".join(actualwords)
                    print(joinedactualwords)

                    
                    lindex = torch.argmax(lm_label_ids[0])
                    maskedword = lm_label_ids[0, lindex].item()
                    maskedword = tokenizer.ids_to_tokens[maskedword]

                    correct = maskedword == realwords[lindex]

                    print(f"The masked word is: {maskedword}, which is {correct}")

                    print("Predicted words:")
                    joinedrealwords = " ".join(realwords)
                    print(joinedrealwords)
                    

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    global_step += 1

                epochloss = tr_loss / nb_tr_steps
                print(f"Current Final Loss: {epochloss}")
                perpl = math.exp(epochloss)
                print(f"Perplexity Final: {perpl}")
                accuracy = right / (right + wrong)
                print(f"Epoch Accuracy: {accuracy}")
                print(f"Total Examples: {totalcounteri}")
                print(f"evaluated Examples: {right + wrong}")

            # Save a trained model
            
##################################################################################
if __name__ == "__main__":
    main()

    print("DONE")