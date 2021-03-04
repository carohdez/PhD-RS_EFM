# Detect aspect category, sentence level

from keras.preprocessing.sequence import pad_sequences
import operator
import math

domain='hotels'

if domain=='restaurant':
  aspects={'ambience':0, 'anecdotes/miscellaneous':1 , 'food':2, 'price':3, 'service':4}
else:
  #aspects={'facilities':0, 'staff':1, 'room':2, 'bathroom':3, 'location':4, 'price':5, 'ambience':6, 'food':7, 'comfort':8, 'cleanliness':9, 'checking':10}
  aspects={'facilities':0, 'staff':1, 'room':2, 'bathroom':3, 'location':4, 'price':5, 'ambience':6, 'food':7, 'comfort':8, 'checking':9}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

prob_threshold=0.7

def get_category(in_text):
    # in_text='the hotel is very close to all main places of interest'
    # in_text='the hotel is very close to all attractions'
    MAX_LEN = 120
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
    categories = []
    for i in range(0, 10):
        index, value = max(enumerate(logits_), key=operator.itemgetter(1))
        if value > 0:
            #print("Aspect " + str(i) + ": " + list(aspects.keys())[list(aspects.values()).index(index)] + ", logit: " + str(value) + ", prob: " + str(format(sigmoid(value), ".2f")))
            if sigmoid(value) >= prob_threshold:
                categories.append(list(aspects.keys())[list(aspects.values()).index(index)])
        logits_[index] = 0
    return(categories)
