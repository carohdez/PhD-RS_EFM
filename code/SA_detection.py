# Detect sentiment, sentence level

from keras.preprocessing.sequence import pad_sequences
import operator
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

prob_threshold=0.8

def get_sentiment(in_text):
    # in_text='the hotel is very close to all main places of interest'
    # in_text='the hotel is very close to all attractions'
    MAX_LEN = 70
    input_ids = tokenizer.encode(in_text, add_special_tokens=True, max_length=MAX_LEN)
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    input_ids = results[0]

    # create attention masks
    attn_mask = [int(i > 0) for i in input_ids]

    # cast to tensor
    input_ids = torch.tensor(input_ids)
    attn_mask = torch.tensor(attn_mask)
    # add an extra dim for the "batch
    input_ids = input_ids.unsqueeze(0)
    attn_mask = attn_mask.unsqueeze(0)

    # BERT Model
    # set model in evaluation mode (dropout layers behave differently during evaluation)
    model.eval()

    # copy inputs to device
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=attn_mask)

    logits_ = logits[0].detach().cpu().numpy()[0]
    sentiment_pos=[i for i, j in enumerate(logits_) if j == max(logits_)][0]
    if sentiment_pos == 0: sentiment="negative"
    elif sentiment_pos == 1: sentiment="positive"
    else: sentiment="neutral"

    return sentiment
    # if logits_[0]>logits_[1]:
    #   return "negative"
    # else:
    #   return "positive"
