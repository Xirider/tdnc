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

#from lambadatest import LambadaTest

from modeling import BertForMaskedLM, BertConfig, BertForMaskedLMUt, UTafterBert, TDNCafterBert
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear

from squaddata import SquadTrain
from wikitextlm import WikitextTrain, prepare_wikitext

from putils import maybe_download, load_weights_from_state

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
_TMPDIR = _CURPATH / "squad_data"
_TRAINDIR = _TMPDIR / "squad_train"
_TESTDIR = _TMPDIR / "squad_test"
_TESTFILE = "dev-v2.0.json"
_DATADIR = _CURPATH / "squad_data"
_TRAINFILE = "train-v2.0.json"
_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/" + _TRAINFILE
_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/" + _TESTFILE
_MODELS = _CURPATH / "models"


_WIKIFOLD = "wikitext-103-raw"
_WIKIFOLDER =  _CURPATH / _WIKIFOLD
_TRAINDIRWIKI = _WIKIFOLDER / "wiki_train"
_DEVDIRWIKI = _WIKIFOLDER / "wiki_dev"
_WIKIZIPFILE = "wikitext-103-raw-v1.zip"
_WIKIURL = "https://s3.amazonaws.com/research.metamind.io/wikitext/" + _WIKIZIPFILE
_WIKITRAINTOKENS = "wiki.train.raw"
_WIKIDEVTOKENS = "wiki.valid.raw"

_EMA_ALPHA = 0.005











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
    parser.add_argument("--max_comp_length",
                        default=256,
                        type=int,
                        help="Maximum amount of tokens in the Ut transformer after the DNC reads tokens from memory")

    parser.add_argument("--memory_size",
                        default=512,
                        type=int,
                        help="DNC memory size")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
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
    parser.add_argument('--direct_write',
                        action='store_true',
                        help="whether to directly write the attention tokens to the dnc memory, or first use an linear transformation")
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
    parser.add_argument("--read_gate",
                action='store_true',
                help="whether to use read gate")
    parser.add_argument("--calc_with_read",
                action='store_true',
                help="whether to use read gate")
    parser.add_argument("--tensorboard",
                action='store_true',
                help="whether to track weights and memories in tensorboard")
    parser.add_argument("--read_token_type", default="concat", type=str, required=False,
                        help="The read tokens can be either concat, added or added and scaled to the original tokens")
    parser.add_argument("--resume",
                action='store_true',
                help="resume from last checkpoint with the same folder name")
    parser.add_argument('--fake_context',
                    type=int,
                    default=0,
                    help="how many in article distractor paragraphs are input")
    parser.add_argument('--dropout',
                        type = float, default = 0.1,
                        help = "Dropout in bert and in DNC")
    parser.add_argument("--task", default="wikitext", type=str, required=False,
                        help="Either squad or wikitext task mode")
    parser.add_argument('--reset_step',
                type=int,
                default=2,
                help="for wikitext, after how many steps to repackage hidden")
    parser.add_argument("--train_full",
                action='store_true',
                help="train the full network instead of just the Ut part")
    args = parser.parse_args()


    hyperparams = args.__dict__

    experiment.log_parameters(hyperparams)

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()


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


    if args.task == "squad":
        if args.download and args.do_train:
            filename = maybe_download(_TRAINDIR, _TRAINFILE, _URL)
        if args.download and args.do_eval:
            filename = maybe_download(_TESTDIR, _TESTFILE, _DEV_URL)
    elif args.task == "wikitext":
        if args.download:
            filename = maybe_download(_WIKIFOLDER, _WIKIZIPFILE, _WIKIURL)
            prepare_wikitext(_WIKIZIPFILE, _WIKIFOLDER, _TRAINDIRWIKI, _DEVDIRWIKI, _WIKITRAINTOKENS, _WIKIDEVTOKENS, _WIKIFOLD)


    lower_case = not args.do_upper_case
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case= lower_case)
    mask_token_number = tokenizer.vocab["[MASK]"]
    # first ist pretrained bert, second is ut following
    config = BertConfig(30522)
    config2 = BertConfig(30522, num_hidden_layers= args.ut_layers, mask_token_number=mask_token_number, 
                            max_comp_length = args.max_comp_length, memory_size = args.memory_size, direct_write =args.direct_write, 
                            read_gate=args.read_gate, read_token_type=args.read_token_type, calc_with_read=args.calc_with_read, hidden_dropout_prob = args.dropout)

    # to test without ut embeddings: , use_mask_embeddings=False, use_temporal_embeddings=False
    

    num_train_optimization_steps = None

    if args.do_train:

        if args.task == "squad":
            train_dataset = SquadTrain(_TRAINDIR, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild, short_factor= args.short_factor, fake_context = args.fake_context)
        elif args.task == "wikitext":
            train_dataset = WikitextTrain(_TRAINDIRWIKI, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild, short_factor= args.short_factor, batch_size = args.train_batch_size // args.gradient_accumulation_steps)

        # corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None,  rebuild=True
        #         , short_factor = 1, distribute_context_over = 1, fake_context=0, out_doc_mult = 1

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

        if args.model_type == "TDNCafterBertPretrained":
            model = TDNCafterBert(config, config2)
        if args.model_type == "UTafterBertPretrained":
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

        model = False
        inter_model = False
        bert_state_dict= False
        cls_state_dict=False




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

    if args.model_type == "TDNCafterBertPretrained":

        model = TDNCafterBert(config, config2)


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
        model.load_state_dict(torch.load(_MODELS / args.load_model / "pytorch_model.pt"), strict=False)
        model.train()

        if args.model_type == "UTafterBertPretrained":
            model.bert.eval()
            model.ut.train()
            model.cls.eval()

            if args.cls_train:
                model.cls.train()

        if args.model_type == "TDNCafterBertPretrained":
            model.bert.eval()
            model.ut.train()
            model.cls.eval()

            if args.cls_train:
                model.cls.train()

    if args.resume:
        print("IMPORTANT: Loaded from last checkpoint")
        model.load_state_dict(torch.load(_MODELS / args.output_dir / "checkpoint.pt"), strict=False)
        model.train()

        if args.model_type == "UTafterBertPretrained":
            model.bert.eval()
            model.ut.train()
            model.cls.eval()

            if args.cls_train:
                model.cls.train()

        if args.model_type == "TDNCafterBertPretrained":
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
    if args.model_type == "UTafterBertPretrained" or "TDNCafterBertPretrained":
        param_optimizer = list(model.ut.named_parameters())
        print("updating only ut part")
        if args.cls_train:
            param_optimizer.extend(list(model.cls.named_parameters()))
            "updating also cls part"
    if args.train_full:
        param_optimizer = list(model.named_parameters())
        print("updating all parameters")

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

        if args.task == "squad":
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_dataset)
            else:
                #TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        elif args.task == "wikitext":
            datalen = len(train_dataset) // args.max_seq_length
            train_dataloader = range(datalen)



        if args.tensorboard:
            model.ut.encoder.layer.inter_results = args.inter_results
            model.ut.encoder.layer.tensorboard = writer


        

        for epochnumber in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            loss_ema = 0
            acc_ema = 0
            best_acc_ema = 0

            lendata =len(train_dataloader)

            if args.task == "wikitext":
                train_dataset.pos = 0

            if args.model_type == "UTafterBertPretrained":
                model.bert.eval()
                model.ut.train()
                model.cls.eval()

                if args.cls_train:
                    model.cls.train()

            if args.model_type == "TDNCafterBertPretrained":
                model.bert.eval()
                model.ut.train()
                model.cls.eval()

                if args.cls_train:
                    model.cls.train()

            if args.train_full:
                model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                if step == lendata or step == lendata-1:
                    break

                if args.tensorboard:
                    model.ut.encoder.layer.outer_steps = step

                if args.task == "squad":
                    context_example_list, question_example = batch
                elif args.task == "wikitext":
                    context_example_list, question_example = train_dataset.get_batch()



                question_example = tuple(t.to(device) for t in question_example)
                context_example_list = [tuple(t.to(device) for t in context_example) for context_example in context_example_list]

                # context
                for contextid, context in enumerate(context_example_list):
                    if contextid == 0:
                        reset_memory = True
                        erase_memory = True
                    else:
                        reset_memory = False
                        erase_memory = False

                    input_ids, input_mask, segment_ids, lm_label_ids  = context
                    _, _ = model(input_ids, segment_ids, input_mask, lm_label_ids , reset_memory=reset_memory, erase_memory=erase_memory)

                if args.task == "squad":
                    q_reset = False
                    q_erase = False
                elif args.task == "wikitext":
                    if step % args.reset_step == 0:
                        accumulated_loss = 0
                        q_reset = True
                        q_erase = False
                        prob = random.random()
                        if prob > 0.995:
                            q_erase = True

                    else:
                        q_reset = False
                        q_reset = False



                # question and answer
                input_ids, input_mask, segment_ids, lm_label_ids  = question_example

                if args.model_type == "UTafterBertPretrained":
                    loss, predictions = model(input_ids, segment_ids, input_mask, lm_label_ids)
                else:
                    loss, predictions = model(input_ids, segment_ids, input_mask, lm_label_ids, reset_memory=q_reset, erase_memory=q_erase)

                if args.task == "wikitext":
                    accumulated_loss += loss


                if args.task == "wikitext" and step % args.reset_step == (args.reset_step - 1):
                    if args.gradient_accumulation_steps > 1:
                        accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
                    accumulated_loss =  accumulated_loss / args.reset_step
                    accumulated_loss.backward()

                if args.task == "squad":
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
            
                losscpu = loss.item()
                print(f"Step {global_step} loss: {losscpu} ")
                experiment.log_metric("loss", losscpu, step = global_step)
                loss_ema = (_EMA_ALPHA * losscpu) + (1.0 - _EMA_ALPHA) * loss_ema
                experiment.log_metric("loss_ema", loss_ema , step = global_step)
                tr_loss += losscpu
            
                with torch.no_grad():

                    maxes = torch.argmax(predictions, 2)
                    correct_number = (lm_label_ids == maxes).sum()
                    correct_number = correct_number.item()
                    totalmasks = (lm_label_ids > 0).sum()
                    totalmasks = totalmasks.item()
            
                cur_accuracy = correct_number / totalmasks
                experiment.log_metric("accuracy", cur_accuracy , step = global_step)
                acc_ema = (_EMA_ALPHA * cur_accuracy) + (1.0 - _EMA_ALPHA) * acc_ema
                experiment.log_metric("accuracy_ema", acc_ema , step = global_step)


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
                    # print("memory of model:")
                    if args.tensorboard:
                        for name, param in model.ut.named_parameters():
                            writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
                        
                        # writer.add_histogram(model.ut.encoder.layer.memory_hidden["memory"][0].abs().sum(1))
                    if args.model_type == "TDNCafterBertPretrained": 
                        print(model.ut.encoder.layer.memory.saved_read_strength[0].mean(0))
                        print("Softmax distribution over 5 %")
                        print((model.ut.encoder.layer.memory.saved_read_softmax > 0.05).sum())
                        print("Softmax distribution over 50 %")
                        print((model.ut.encoder.layer.memory.saved_read_softmax > 0.5).sum())
                        print("Softmax distribution over 99 %")
                        print((model.ut.encoder.layer.memory.saved_read_softmax > 0.99).sum())
                        print("Softmax distribution over 99,999 %")
                        print((model.ut.encoder.layer.memory.saved_read_softmax > 0.99999).sum())


                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                global_step += 1
                if args.task == "squad" or step % args.reset_step == (args.reset_step - 1):

                    if step % args.gradient_accumulation_steps == 0:
                        # if args.fp16:
                        #     # modify learning rate with special warm up BERT uses
                        #     # if args.fp16 is False, BertAdam is used that handles this automatically
                        #     lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        #     hyperparams["learning_rate"] = lr_this_step
                        #     for param_group in optimizer.param_groups:
                        #         param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        experiment.log_metric("current_lr", optimizer.current_lr , step = global_step)
                        

                        if step  % 2500 == 1:
                            if acc_ema > best_acc_ema:
                                best_acc_ema = acc_ema
                            if not os.path.exists(_MODELS):
                                os.makedirs(_MODELS)
                            if not os.path.exists(_MODELS/ args.output_dir):
                                os.makedirs( _MODELS / args.output_dir)
                            output_model_file = os.path.join(_MODELS , args.output_dir, "checkpoint.pt")
                            torch.save(model.state_dict(), output_model_file)
                            logger.info(f"Created new checkpoint")

                # here start eval each epoch


                    
            if args.do_eval:  

                if args.task == "squad":
                    test_dataset = SquadTrain(_TESTDIR, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild, short_factor= args.short_factor, fake_context = args.fake_context)
                    test_sampler = RandomSampler(test_dataset)
                    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)
                elif args.task == "wikitext":
                    test_dataset = WikitextTrain(_TRAINDIRWIKI, tokenizer, seq_len = args.max_seq_length, rebuild=args.rebuild, short_factor= args.short_factor, batch_size = args.train_batch_size // args.gradient_accumulation_steps, variable_seq = False)
                    datalen = len(test_dataset)
                    test_dataloader = range(datalen)
                    test_dataset.pos = 0
                model.eval()
                with torch.no_grad():
                    for _ in trange(1, desc="Epoch"):
                        test_loss = 0
                        nb_test_examples, nb_test_steps = 0, 0
                        totalcounteri = 0
                        total_acc = 0
                        lentest = len(test_dataloader)
                        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
                            if step == lentest or step == lentest-1:
                                break
                            

                            context_example_list, question_example = batch
                            question_example = tuple(t.to(device) for t in question_example)
                            context_example_list = [tuple(t.to(device) for t in context_example) for context_example in context_example_list]

                        # context
                            for contextid, context in enumerate(context_example_list):
                                if contextid == 0:
                                    reset_memory = True
                                    erase_memory = True
                                else:
                                    reset_memory = False
                                    erase_memory = False

                                input_ids, input_mask, segment_ids, lm_label_ids  = context

                                _, _ = model(input_ids, segment_ids, input_mask, lm_label_ids , reset_memory=reset_memory, erase_memory=erase_memory)

                        
                        # question and answer
                            input_ids, input_mask, segment_ids, lm_label_ids  = question_example
                            loss, predictions = model(input_ids, segment_ids, input_mask, lm_label_ids, reset_memory=False, erase_memory=False)





                            if n_gpu > 1:
                                loss = loss.mean() # mean() to average on multi-gpu.
                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps

                            maxes = torch.argmax(predictions, 2)
                            correct_number = (lm_label_ids == maxes).sum()
                            correct_number = correct_number.item()
                            totalmasks = (lm_label_ids > 0).sum()
                            totalmasks = totalmasks.item()
                            if totalmasks > 0:
                                cur_accuracy = correct_number / totalmasks
                                total_acc += cur_accuracy

                            print(f"Current Loss: {loss.item()}")

                            perpl = math.exp(loss.item())
                            print(f"Perplexity: {perpl}")
                            test_loss += loss.item()
                            nb_test_steps += 1

                    epochloss = test_loss / nb_test_steps
                    print(f"Loss for test Data: {epochloss}")
                    experiment.log_metric("test_loss", epochloss, step = epochnumber)
                    perpl = math.exp(epochloss)
                    print(f"Perplexity for Test Data: {perpl}")
                    experiment.log_metric("test_perplexity", perpl, step = epochnumber)
                    accuracy = total_acc / nb_test_steps
                    print(f"Accuracy for Test Data: {accuracy}")
                    experiment.log_metric("test_accuracy", accuracy, step = epochnumber)



                
        if args.tensorboard:
            writer.close()
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


##################################################################################
if __name__ == "__main__":
    main()

    print("DONE")