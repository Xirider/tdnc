



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






class SquadTrain(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None,  rebuild=True
                , short_factor = 1, distribute_context_over = 1, fake_context=0, out_doc_mult = 1 ):

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
        self.short_factor = short_factor

        self.distribute_context_over = distribute_context_over
        self.fake_context = fake_context
        self.out_doc_mult = out_doc_mult

        self.answer_size = 5
        self.qa_extra_tokens = 4 # Question: Answer:

        self.max_questions = 1


        if rebuild:
            self.docs = []
            skipcounter = 0
            if not os.path.exists(self.corpus_path):
                os.makedirs(self.corpus_path)

            if os.path.exists(self.corpus_path / "build_docs_train.p"):
                os.remove(self.corpus_path / "build_docs_train.p")
            # for subdir in os.listdir(self.corpus_path):
            #     print(subdir)
            #     for files in os.listdir(self.corpus_path / subdir):
            #         with open(self.corpus_path / subdir / files , "r", encoding=encoding) as f:
            #             interdoc = ""

            #             for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                            

            #                 interdoc += line

            #                 if len(interdoc.split()) >= self.creation_length:
            #                     if skipcounter % self.short_factor == 0:
            #                         self.docs.append(interdoc)
            #                         self.num_docs += 1
            #                     interdoc = ""
            #                     skipcounter += 1
                            
            #     print("genre done")
            
            import json
            files = os.listdir(self.corpus_path)[0]
                
            with open(self.corpus_path/files, "r", encoding=self.encoding) as json_file:
                data_dict = json.load(json_file)

            data_dict = data_dict["data"]
            number_articles = len(data_dict)
            total = 0
            context_counter = 0
            counter_del = 0
            self.data = []
            
            # remove all too long paragraphs
            for article in range(number_articles):
                cur_number_context = len(data_dict[article]["paragraphs"])
                # print(f"This is article number {article}")
                # print(cur_number_context)
                cont_this_article = 0
                for context in range(cur_number_context -1, -1, -1):
                    
                    context_counter += 1
                    context_string = data_dict[article]["paragraphs"][context]["context"]
                    tokens = self.tokenizer.tokenize(context_string)
                    context_len = len(tokens)
                    if context_len > (self.seq_len - 2):
                        del data_dict[article]["paragraphs"][context]
                        counter_del += 1
                    else:
                        self.data.append({ "data" : data_dict[article]["paragraphs"][context], "article_number": article, "bw_context_number": cont_this_article })
                        cont_this_article += 1
            
            self.data.reverse()
            self.data_dict = data_dict
            
            print(f"Deleted {counter_del} too short contexts of {context_counter} total contexts")
            
            packed_data = {"data_save" : self.data, "data_dict": self.data_dict}

            pickle.dump(packed_data, open(self.corpus_path / "build_docs_train.p", "wb"))
            print("Saved Dataset with Pickle")
        
        else:
            packed_data = pickle.load( open(self.corpus_path / "build_docs_train.p", "rb"))

            self.data = packed_data["data_save"]
            self.data_dict = packed_data["data_dict"]

            print("Loaded Dataset with Pickle")


    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return len(self.data)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        
        example = self.data[item]
        data_example = example["data"]
        article_number = example["article_number"]
        bw_context_number = example["bw_context_number"]
        # get the context and tokenize it
        true_context = self.tokenizer.tokenize(data_example["context"])

        # sample fake context, except for out of doc context
        cur_article = self.data_dict[article_number]["paragraphs"]
        number_paragraphs = len(cur_article)
        fw_context_number = number_paragraphs - bw_context_number - 1
        assert (fw_context_number > -0.5)
        context_list = [true_context]


        # insert_pos = random.randrange(0, ((self.fake_context + 1)* self.out_doc_mult)+1)
        already_drawn = [fw_context_number]

        for fakes in range(self.fake_context):
            draw = fw_context_number
            while draw in already_drawn:
                draw = random.randrange(0, number_paragraphs)
            
            already_drawn.append(draw)
            drawn_para = cur_article[draw]["context"]
            context_list.append(self.tokenizer.tokenize(drawn_para))

        random.shuffle(context_list)

        # sample out of doc context

        # get the questions, tokenize them, randomly sample enough to fit seq length, put them into a list of lists 
        question_dict_list = data_example["qas"]
        question_list = []
        answer_list = []
        qa_len = 0
        random.shuffle(question_dict_list)
        #print(question_dict_list)
        #import pdb; pdb.set_trace()
        for question in question_dict_list:
            question_tokens = self.tokenizer.tokenize(question["question"])
            is_impossible = question["is_impossible"]
            if not is_impossible:
                answer_tokens = self.tokenizer.tokenize(question["answers"][0]["text"])
            else:
                answer_tokens = []

            if (len(question_tokens) + self.answer_size + self.qa_extra_tokens + qa_len ) < (self.seq_len - 2):
                if not is_impossible and len(answer_tokens) <= self.answer_size :
                    question_list.append(question_tokens)
                    answer_list.append(answer_tokens)
                    qa_len += len(question_tokens) + self.answer_size + self.qa_extra_tokens
   
            else:
                break

            if len(question_list) == self.max_questions:
                break




        # transform sample to features
        context_example_list, question_example = convert_example_to_features(context_list, question_list, answer_list, self.sample_counter, self.seq_len, self.tokenizer, self.answer_size)

        # cur_tensors = (torch.tensor(cur_features.input_ids),
        #                torch.tensor(cur_features.input_mask),
        #                torch.tensor(cur_features.segment_ids),
        #                torch.tensor(cur_features.lm_label_ids),
        #                )

        return [input_example.release_features() for input_example in context_example_list] , question_example.release_features()






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

    def release_features(self):
        cur_tensors = (torch.tensor(self.input_ids),
                        torch.tensor(self.input_mask),
                        torch.tensor(self.segment_ids),
                        torch.tensor(self.lm_label_ids))
        return cur_tensors


def convert_example_to_features(context_list, question_list, answer_list, cur_time, max_seq_length, tokenizer, answer_size):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"

    #import pdb; pdb.set_trace()



    for contexts in context_list:
        contexts.insert(0, "[CLS]")
        contexts.append("[SEP]")


    question_text = ["[CLS]"]
    label_text = [-1]
    for qid, question in enumerate(question_list):
        question_text.append("question")
        question_text.append(":")

        question_text.extend(question)
        label_text.extend([-1] * (len(question) + 4))

        question_text.append("answer")
        question_text.append(":")

        question_text.extend(["[MASK]"]* answer_size)

        answer_ids = tokenizer.convert_tokens_to_ids(answer_list[qid])
        label_text.extend(answer_ids)
        label_text.extend([tokenizer.vocab["."]]*(answer_size - len(answer_list[qid])))

    
    question_text.append("[SEP]")
    saved_tokens_question = question_text.copy()
    label_text.append(-1)

    question_text = tokenizer.convert_tokens_to_ids(question_text)
    context_list = [tokenizer.convert_tokens_to_ids(x) for x in context_list]

    # lm_label_ids = ([-1] + t1_label + [-1])

    # tokens = []
    # segment_ids = []
    # tokens.append("[CLS]")
    # segment_ids.append(0)

    # for token in tokens_a:
    #     tokens.append(token)
    #     segment_ids.append(0)

    # tokens.append("[SEP]")
    # segment_ids.append(0)


    q_input_mask = [1] * len(question_text)
    q_segment_ids = [0] * len(question_text)
    q_lm_label_ids = label_text
    while len(question_text) < max_seq_length:
        question_text.append(0)
        q_input_mask.append(0)
        q_segment_ids.append(0)
        q_lm_label_ids.append(-1)
    
    assert len(question_text) == max_seq_length
    assert len(q_input_mask) == max_seq_length
    assert len(q_segment_ids) == max_seq_length
    assert len(q_lm_label_ids) == max_seq_length


    question_example = InputFeatures(input_ids = question_text, input_mask = q_input_mask, segment_ids = q_segment_ids, lm_label_ids = q_lm_label_ids)
    
    context_example_list = []
    for context in context_list:

        c_input_mask = [1] * len(context)
        c_segment_ids = [0] * len(context)
        c_lm_label_ids = [-1] * len(context)
        while len(context) < max_seq_length:
            context.append(0)
            c_input_mask.append(0)
            c_segment_ids.append(0)
            c_lm_label_ids.append(-1)
        

        assert len(context) == max_seq_length
        assert len(c_input_mask) == max_seq_length
        assert len(c_segment_ids) == max_seq_length
        assert len(c_lm_label_ids) == max_seq_length

        context_example = InputFeatures(input_ids = context, input_mask = c_input_mask, segment_ids = c_segment_ids, lm_label_ids = c_lm_label_ids)
        context_example_list.append(context_example)

    #input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)

    # # Zero-pad up to the sequence length.
    # while len(input_ids) < max_seq_length:
    #     input_ids.append(0)
    #     input_mask.append(0)
    #     segment_ids.append(0)
    #     lm_label_ids.append(-1)

    # print("input, segment, lmlabel")
    # print(len(input_ids))
    # print(len(segment_ids))
    # print(len(lm_label_ids))

    # assert len(input_ids) == max_seq_length
    # assert len(input_mask) == max_seq_length
    # assert len(segment_ids) == max_seq_length
    # assert len(lm_label_ids) == max_seq_length

    if cur_time < 5:
        logger.info("*** Example for the Questions***")
        logger.info("cur_time: %s" % (cur_time))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in saved_tokens_question ]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in question_text]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in q_input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in q_segment_ids]))
        logger.info("LM label: %s " % (q_lm_label_ids))

    # features = InputFeatures(input_ids=input_ids,
    #                          input_mask=input_mask,
    #                          segment_ids=segment_ids,
    #                          lm_label_ids=lm_label_ids,
    #                          )
    return context_example_list, question_example


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

