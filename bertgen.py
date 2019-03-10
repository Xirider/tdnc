import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# # Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()
model.to('cuda')

# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
#text = "I drive a car. It has 4 large"
#text = "[CLS] The event consisted of eight professional wrestling matches with wrestlers involved in pre-existing scripted feuds, and storylines. Wrestlers were portrayed as either villains or fan [MASK] . [SEP]"
text = "The battery on Logan’s radio must have been on the way out. So he told himself. There was no other explanation beyond Cygan and the staff at the White House having been overrun. Lizzie opened her eyes with a flutter. Car . Car . Car . Car.  They had been on the icy road for an hour without incident. Jack was happy to do all of the"
#text = "Sophie Le Vinh ( born January 25 , 1994 ) is a german artist. She is best known for her watercolor painting"
savedtext = tokenizer.tokenize(text)

print(len(savedtext))

#print(tokenized_text)
number = 30
temperature = 100

for t in range(number):

    #import pdb; pdb.set_trace()
    tokenized_text = savedtext.copy()

    tokenized_text.extend(["[MASK]",".","[SEP]"])
    tokenized_text.insert(0, "[CLS]")
    #print(tokenized_text)
    # masked_index = 7
    # tokenized_text[masked_index] = "[MASK]"

    #assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #indexed_tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)
    #print(indexed_tokens)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    #segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    segments_ids = [0] * len(tokenized_text)

    # # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])



    # # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')


    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

        predictions = predictions.squeeze(0)
        words = torch.chunk(predictions, predictions.size(0))
    # # confirm we were able to predict 'henson'
    #words = [torch.argmax(x).item() for x in words]
    #word = words[-3]

        words = [torch.multinomial(torch.pow(temperature, x),1).item() for x in words]

    realwords = tokenizer.convert_ids_to_tokens(words)
    realword = realwords[-3]
    print("raw output:")
    print(realword)
    savedtext.append(realword)

print("Final Answer: ")
print(savedtext)
