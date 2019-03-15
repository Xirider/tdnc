# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from comet_ml import Experiment


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

from lambadatest import LambadaTest

from modeling import BertForMaskedLM, BertConfig, BertForMaskedLMUt, UTafterBert
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear

import random

import tarfile
import requests

experiment = Experiment(api_key="zMVSRiUzF89hdX5u7uWrSW5og",
                        project_name="general", workspace="xirider")


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




_CURPATH = Path.cwd() 
_TMPDIR = _CURPATH / "labada_data_intermediate"
_TRAINDIR = _TMPDIR / "train-novels"
_TESTFILE = "lambada_development_plain_text.txt"
_DATADIR = _CURPATH / "labada_data"
_TAR = "lambada-dataset.tar.gz"
_URL = "http://clic.cimec.unitn.it/lambada/" + _TAR
_MODELS = _CURPATH / "models"

_EMA_ALPHA = 0.025



def maybe_download(directory, filename, uri):
  
    filepath = os.path.join(directory, filename)
    if not os.path.exists(directory):
        logger.info(f"Creating new dir: {directory}")
        os.makedirs(directory)
    if not os.path.exists(filepath):
        logger.info("Downloading und unpacking file, as file does not exist yet")
        r = requests.get(uri, allow_redirects=True)
        open(filepath, "wb").write(r.content)

    return filepath
    


def _prepare_lambada_data(tmp_dir, data_dir):

    file_path = maybe_download(tmp_dir, _TAR, _URL)
    tar_all = tarfile.open(file_path)
    tar_all.extractall(tmp_dir)
    tar_all.close()
    tar_train = tarfile.open(os.path.join(tmp_dir, "train-novels.tar"))
    tar_train.extractall(tmp_dir)
    tar_train.close()

    return None





class LambadaTrain(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None,  rebuild=True, creation_length =128, short_factor = 1):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        self.creation_length = creation_length

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.short_factor = short_factor



        if rebuild:
            self.docs = []
            skipcounter = 0
            if os.path.exists(self.corpus_path / "build_docs.p"):
                os.remove(self.corpus_path / "build_docs.p")
            for subdir in os.listdir(self.corpus_path):
                print(subdir)
                for files in os.listdir(self.corpus_path / subdir):
                    with open(self.corpus_path / subdir / files , "r", encoding=encoding) as f:
                        interdoc = ""

                        for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                            

                            interdoc += line

                            if len(interdoc.split()) >= self.creation_length:
                                if skipcounter % self.short_factor == 0:
                                    self.docs.append(interdoc)
                                    self.num_docs += 1
                                interdoc = ""
                                skipcounter += 1
                            
                print("genre done")
            
            pickle.dump(self.docs, open(self.corpus_path / "build_docs.p", "wb"))
            print("Saved Dataset with Pickle")
        
        else:
            self.docs = pickle.load( open(self.corpus_path / "build_docs.p", "rb"))
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




    tokens_a = truncate_seq_pair(tokens_a, max_seq_length - 2)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)


    lm_label_ids = ([-1] + t1_label + [-1])

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
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
            tokens_a.pop()
    return tokens_a



def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    # changed to always remove 15% of words

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            tokens[i] = "[MASK]"

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label




def load_weights_from_state(model, state_dict):

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    start_prefix = ''
    if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
        start_prefix = 'bert.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            model.__class__.__name__, "\n\t".join(error_msgs)))
    return model




















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
                        default=64,
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
    parser.add_argument("--do_upper_case",
                        action='store_false',
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
    parser.add_argument('--inter_results',
                        type=int,
                        default=100,
                        help="how often to give the results")
    parser.add_argument('--short_factor',
                        type=int,
                        default=1,
                        help="divide training set length by factor")
    parser.add_argument("--download",
                    action='store_true',
                    help="Whether to download the data again")
    parser.add_argument("--rebuild",
                    action='store_true',
                    help="whether to process the data again")
    parser.add_argument("--model_type", default="bert", type=str, required=False,
                        help="Instead of google pretrained models use another model")
    parser.add_argument("--copy_google_weights",
                    action='store_true',
                    help="Whether to copy and save a new version of Bert weights from the google weights")
    parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval")
    parser.add_argument('--ut_layers',
                type=int,
                default=4,
                help="layers for ut model")
    parser.add_argument("--load_model", default="", type=str, required=False,
                    help="Load model in corresponding Model folder")
    parser.add_argument("--cls_train",
                action='store_true',
                help="Whether to train the cls layer")

    args = parser.parse_args()


    hyperparams = args.__dict__

    experiment.log_parameters(hyperparams)

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

    if args.download and args.do_train:
        filename = _prepare_lambada_data(_TMPDIR, _DATADIR)

    lower_case = not args.do_upper_case
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case= lower_case)

    mask_token_number = tokenizer.vocab["[MASK]"]
    # first ist pretrained bert, second is ut following
    config = BertConfig(30522)
    config2 = BertConfig(30522, num_hidden_layers= args.ut_layers, mask_token_number=mask_token_number)

    # to test without ut embeddings: , use_mask_embeddings=False, use_temporal_embeddings=False
    

    num_train_optimization_steps = None

    if args.do_train:

        train_dataset = LambadaTrain(_TRAINDIR, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild, creation_length= args.max_seq_length - 20, short_factor= args.short_factor)

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

    

    if args.copy_google_weights:
        inter_model = BertForMaskedLM.from_pretrained(args.bert_model)
        bert_state_dict = inter_model.bert.state_dict()
        cls_state_dict = inter_model.cls.state_dict()

        model = UTafterBert(config, config2)

        # state = model.cls.state_dict()
        # for name in state:
        #     print(name)
        #     print(state[name])

        load_weights_from_state(model.bert, bert_state_dict)
        load_weights_from_state(model.cls , cls_state_dict)

        if not os.path.exists(_MODELS):
            logger.info(f"Creating new dir: {_MODELS}")
            os.makedirs(_MODELS)

        torch.save(model.state_dict(), _MODELS / "UTafterBertPretrained.pt")




    # prepare model:
    if args.model_type == "bert":
        model = BertForMaskedLM.from_pretrained(args.bert_model)
        model.train()
    
    if args.model_type == "bert_base_untrained":
        model = BertForMaskedLM(config)
        model.train()

    if args.model_type == "bert_base_ut_untrained":
        model = BertForMaskedLMUt(config)
        model.train()

    if args.model_type == "bert_base_ut_after_bert_untrained":
        model = UTafterBert(config, config2)
        model.train()

    if args.model_type == "UTafterBertPretrained":

        model = UTafterBert(config, config2)


        model.load_state_dict(torch.load(_MODELS / "UTafterBertPretrained.pt"))

        model.bert.eval()
        model.ut.train()
        model.cls.eval()

        if args.cls_train:
            model.cls.train()

        
        for param in model.bert.parameters():
            param.requires_grad = False

    if args.load_model != "":
        print("Load model saved model")
        model.load_state_dict(torch.load(_MODELS / args.load_model / "pytorch_model.pt"))
        model.train()

        if args.model_type == "UTafterBertPretrained":
            model.bert.eval()
            model.ut.train()
            model.cls.eval()

            if args.cls_train:
                model.cls.train()



    if args.fp16:
        model.half()
    model.to(device)
    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #     model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.model_type == "UTafterBertPretrained":
        param_optimizer = list(model.ut.named_parameters())
        print("updating only ut part")
        if args.cls_train:
            param_optimizer.extend(list(model.cls.named_parameters()))

    else:
        param_optimizer = list(model.named_parameters())
        print("updating all parameters")
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # if args.fp16:
    #     try:
    #         from apex.optimizers import FP16_Optimizer
    #         from apex.optimizers import FusedAdam
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    #     optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False,
    #                           max_grad_norm=1.0)
    #     if args.loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    # else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    

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



        

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            loss_ema = 0
            acc_ema = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids  = batch
                loss, predictions = model(input_ids, segment_ids, input_mask, lm_label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # Metrics
                losscpu = loss.item()
                print(f"Step {step} loss: {losscpu} ")
                experiment.log_metric("loss", losscpu, step = step)
                loss_ema = (_EMA_ALPHA * losscpu) + (1.0 - _EMA_ALPHA) * loss_ema
                experiment.log_metric("loss_ema", loss_ema , step = step)
                tr_loss += losscpu
                
                with torch.no_grad():

                    maxes = torch.argmax(predictions, 2)
                    correct_number = (lm_label_ids == maxes).sum()
                    correct_number = correct_number.item()
                    totalmasks = (lm_label_ids > 0).sum()
                    totalmasks = totalmasks.item()
                
                cur_accuracy = correct_number / totalmasks
                experiment.log_metric("accuracy", cur_accuracy , step = step)
                acc_ema = (_EMA_ALPHA * cur_accuracy) + (1.0 - _EMA_ALPHA) * acc_ema
                experiment.log_metric("accuracy_ema", acc_ema , step = step)


                if step % args.inter_results == 0:

                    with torch.no_grad():
                        predictions = predictions[0]
                        words = torch.chunk(predictions, predictions.size(0))
                        words = [torch.argmax(x).item() for x in words]
                    realwords = tokenizer.convert_ids_to_tokens(words)
                    print("Real sentences:")
                    with torch.no_grad():
                        firstbatch = input_ids[0]
                        words = firstbatch.tolist()

                    actualwords = tokenizer.convert_ids_to_tokens(words)
                    joinedactualwords = " ".join(actualwords)
                    print(joinedactualwords)

                    print("Predicted words:")
                    joinedrealwords = " ".join(realwords)
                    print(joinedrealwords)





                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        hyperparams["learning_rate"] = lr_this_step
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    experiment.log_metric("current_lr", optimizer.current_lr , step = step)
                    global_step += 1
                


        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")


        if args.do_train:
            if not os.path.exists(_MODELS):
                os.makedirs(_MODELS)
            if not os.path.exists(_MODELS/ args.output_dir):
                os.makedirs( _MODELS / args.output_dir)
            output_model_file = os.path.join(_MODELS , args.output_dir, "pytorch_model.pt")
            torch.save(model.state_dict(), output_model_file)
            logger.info(f"Creating new dir and saving model in: {args.output_dir}")


    if args.do_eval:  

        test_dataset = LambadaTest(_TMPDIR, _TESTFILE, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild)
        test_sampler = RandomSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)

        model.eval()
        with torch.no_grad():
            for _ in trange(1, desc="Epoch"):
                test_loss = 0
                nb_test_examples, nb_test_steps = 0, 0
                totalcounteri = 0
                total_acc = 0
                for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, lm_label_ids  = batch
                    loss, predictions = model(input_ids, segment_ids, input_mask, lm_label_ids)
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    maxes = torch.argmax(predictions, 2)
                    correct_number = (lm_label_ids == maxes).sum()
                    correct_number = correct_number.item()
                    totalmasks = (lm_label_ids > 0).sum()
                    totalmasks = totalmasks.item()
                
                    cur_accuracy = correct_number / totalmasks
                    total_acc += cur_accuracy

                    print(f"Current Loss: {loss.item()}")

                    perpl = math.exp(loss.item())
                    print(f"Perplexity: {perpl}")
                    test_loss += loss.item()
                    nb_test_steps += 1

            epochloss = test_loss / nb_test_steps
            print(f"Loss for test Data: {epochloss}")
            experiment.log_metric("test_loss", epochloss)
            perpl = math.exp(epochloss)
            print(f"Perplexity for Test Data: {perpl}")
            experiment.log_metric("test_perplexity", perpl)
            accuracy = total_acc / nb_test_steps
            print(f"Accuracy for Test Data: {accuracy}")
            experiment.log_metric("test_accuracy", accuracy)

##################################################################################
if __name__ == "__main__":
    main()

    print("DONE")