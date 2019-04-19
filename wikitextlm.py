import os
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import random
from io import open
import pickle
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
import time
import torch
import logging
logger = logging.getLogger(__name__)
import numpy as np
import shutil


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

# wiki train 
_WIKIFOLD = "wikitext-103-raw"
_WIKIFOLDER =  _CURPATH / _WIKIFOLD
_TRAINDIRWIKI = _WIKIFOLDER / "wiki_train"
_DEVDIRWIKI = _WIKIFOLDER / "wiki_dev"
_WIKIZIPFILE = "wikitext-103-raw-v1.zip"
_WIKIURL = "https://s3.amazonaws.com/research.metamind.io/wikitext/" + _WIKIZIPFILE
_WIKITRAINTOKENS = "wiki.train"
_WIKIDEVTOKENS = "wiki.valid"

def unzip(filename, targetdir):
    import zipfile
    with zipfile.ZipFile(filename,"r") as zip_ref:
        zip_ref.extractall(targetdir)

def prepare_wikitext(zipfile, wikifolder, traindirwiki, devdirwiki, wikitraintokens, wikidevtokens, wikifold):
    print("unzipping wikifiles")
    unzip(wikifolder / zipfile, wikifolder)
    if not os.path.exists(traindirwiki):
        os.makedirs(traindirwiki)
    if not os.path.exists(devdirwiki):
        os.makedirs(devdirwiki)
    os.rename(wikifolder / wikifold/ wikitraintokens, traindirwiki / wikitraintokens )
    os.rename(wikifolder / wikifold / wikidevtokens, devdirwiki / wikidevtokens )
    print("finished moving wikifolder")


def chunk(l, n):
    length = len(l) / n
    assert (length % 1 == 0)
    length = int(length)
    for i in range(n):
        yield l[i * length: (i+1) * length]


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    del data[nbatch * bsz:]
    assert (len(data) == nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = list(chunk(data, bsz))
    # data = data.view(bsz, -1).t().contiguous()
    return data


def main():
    from tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case= True)
    abc = WikitextTrain(_TRAINDIRWIKI, tokenizer, seq_len = 128 , rebuild = True)


class WikitextTrain(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None,  rebuild=True, short_factor = 1, batch_size = 2, variable_seq=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 10000
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.pos = 0  # to avoid random sentence from same doc
        self.batch_size = batch_size
        self.variable_seq = variable_seq
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
            for files in os.listdir(self.corpus_path):
                with open(self.corpus_path / files , "r", encoding=encoding) as f:
                    
                    print("Calculating length of document")
                    corpus_lines = sum(1 for line in f)
                    print(corpus_lines)
                    counter = 0
                    f.seek(0)


                    for line in tqdm(f, desc="Tokenization", total=corpus_lines):
                        words = line.split()
                        for i ,word in enumerate(words):
                            if word == "@-@":
                                words[i]  = "-"  
                            if word == "@.@":
                                words[i]  = "."  
                            if word == "@,@":
                                words[i]  = ","  
                        words = " ".join(words)



                        split_tokens = self.tokenizer.tokenize(words)
                        tokens = self.tokenizer.convert_tokens_to_ids(split_tokens)
                        self.docs.extend(tokens)
                        counter += 1
                        if counter < 100:
                            print(split_tokens)
                            print(tokens)

            # self.docs = torch.LongTensor(self.docs)


            print("Tokenization of full corpus done")
            print(f"Full number of tokens: {len(self.docs)}")
            
            pickle.dump(self.docs, open(self.corpus_path / "build_docs.p", "wb"))
            print("Saved Dataset with Pickle")
        
        else:
            self.docs = pickle.load( open(self.corpus_path / "build_docs.p", "rb"))
            print("Loaded Dataset with Pickle")

        self.docs = batchify(self.docs, batch_size)

        # self.length = self.docs.size(1)
        self.length = len(self.docs[0])



    # def batchify(data, bsz):
    #     # Work out how cleanly we can divide the dataset into bsz parts.
    #     nbatch = data.size(0) // bsz
    #     # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #     data = data.narrow(0, 0, nbatch * bsz)
    #     # Evenly divide the data across the bsz batches.
    #     data = data.view(bsz, -1).t().contiguous()
    #     return data

    def get_batch(self):
        
        i = self.pos
        # Prevent excessively small or negative sequence lengths
        
        if self.variable_seq:
            seq_len = max(int(self.seq_len *0.5), int(np.random.normal(self.seq_len, self.seq_len *0.2)))
            prob = random.random()
            if prob > 0.97: seq_len = seq_len // 2
            seq_len = min(self.seq_len - 2, seq_len)
        else:
            seq_len = self.seq_len - 2

        data = [ doc[i:i+seq_len] for doc in self.docs]

        self.pos += seq_len
        cur_features = convert_example_to_features(data, self.sample_counter, self.seq_len, self.tokenizer)
        self.sample_counter += 1
        
        cur_tensors = (cur_features.input_ids,
                       cur_features.input_mask,
                       cur_features.segment_ids,
                       cur_features.lm_label_ids,
                       )

        return [], cur_tensors



    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return len(self.docs[0])








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



def convert_example_to_features(batch, cur_time, max_seq_length, tokenizer):
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

    batch_size = len(batch)

    input_ids_tensor = torch.zeros([batch_size, max_seq_length], dtype= torch.long)
    input_mask_tensor = torch.zeros([batch_size, max_seq_length], dtype= torch.long)
    segment_ids_tensor = torch.zeros([batch_size, max_seq_length], dtype= torch.long)
    lm_label_ids_tensor = torch.zeros([batch_size, max_seq_length], dtype= torch.long)

    for b, example in enumerate(batch):

        masked_tokens, labels = random_word(example, tokenizer)


        lm_label_ids = ([-1] + labels + [-1])

        tokens = []
        segment_ids = []
        tokens.append(tokenizer.vocab["[CLS]"])
        segment_ids.append(0)

        for token in masked_tokens:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append(tokenizer.vocab["[SEP]"])
        segment_ids.append(0)


        input_ids = tokens

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
            input_ids_decoded = tokenizer.convert_ids_to_tokens(input_ids)
            logger.info("input_ids decoded: %s" % " ".join([str(x) for x in input_ids_decoded]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("LM label: %s " % (lm_label_ids))


        input_ids_tensor[b] = torch.FloatTensor(input_ids)
        input_mask_tensor[b] = torch.FloatTensor(input_mask)
        segment_ids_tensor[b] = torch.FloatTensor(segment_ids)
        lm_label_ids_tensor[b] = torch.FloatTensor(lm_label_ids)




    features = InputFeatures(input_ids=input_ids_tensor,
                            input_mask=input_mask_tensor,
                            segment_ids=segment_ids_tensor,
                            lm_label_ids=lm_label_ids_tensor,
                            )
    return features




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

            tokens[i] = tokenizer.vocab["[MASK]"]

            # append current token to output (we will predict these later)
            try:
                output_label.append(token)
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

if __name__ == "__main__":
    main()