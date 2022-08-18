from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModel, TFAutoModel, BertForPreTraining, BertForSequenceClassification, BertForTokenClassification
import torch
import numpy as np
import csv

#set up BERT
#BERTje
#https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#22-tokenization
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased", output_hidden_states = True)  # PyTorch

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

#get new sentence
filename = 'all_labeled_data.csv'
df = pd.read_csv(filename, delimiter=';', encoding='utf-8')

mylist = []
myList_current_smok = []
myList_past_smok = []

for index, row in df.iterrows():
    string = row['txt'].lower()
    # find word "roken"
    smoking_index = string.find(" roken")
    if smoking_index == -1:
        mylist.append(string[:40])
        #print("smoking  not found but string is lengeth", len(string[:40]))
        continue
    #get 40 positions before and 50 positions after #this is something I can experiment with
    start = smoking_index - 100
    end = smoking_index + 100
    new_string = string[start:end]
    #print(len(new_string))
    mylist.append(new_string)
    if row['label'] == 'voorheen':
        myList_past_smok.append(new_string)
    else:
        myList_current_smok.append(new_string)


#randomly select 100 items from each list
n = 100
past_smoker_set = random.sample(myList_past_smok, n)
current_smoker_set = random.sample(myList_current_smok, n)
print(past_smoker_set)
fields = ['sentence']

with open('past_smoker_set.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows([[item] for item in past_smoker_set])

with open('current_smoker_set.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows([[item] for item in current_smoker_set])

#apply embedding
embeddings1 = []
# ---Add markers for each sentence
for letter_num in range(len(past_smoker_set)):
    print(letter_num, "out of ", len(past_smoker_set))

    marked_text = "[CLS] " + past_smoker_set[letter_num] + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

        # print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        layer_i = 0

        # print("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0

        # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0

        # print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # `token_vecs` is a tensor with shape [numtokens x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = sentence_embedding.numpy()

    # add letter embedding to embedding
    embeddings1.append(sentence_embedding)

# print(len(embeddings))
ar1 = np.array(embeddings1, dtype=object)
print(np.shape(ar1))



#apply embedding
embeddings2 = []
# ---Add markers for each sentence
for letter_num in range(len(current_smoker_set)):
    print(letter_num, "out of ", len(current_smoker_set))

    marked_text = "[CLS] " + current_smoker_set[letter_num] + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

        # print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        layer_i = 0

        # print("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0

        # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0

        # print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # `token_vecs` is a tensor with shape [numtokens x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = sentence_embedding.numpy()

    # add letter embedding to embedding
    embeddings2.append(sentence_embedding)

# print(len(embeddings))
ar2 = np.array(embeddings2, dtype=object)
print(np.shape(ar2))

sim_past_current = []
sim_past_past = []
sim_current_current = []

for i in range(len(past_smoker_set)):
    for j in range(len(current_smoker_set)):
        #calculate cosin sim between the two sentences
        sentence1 = ar1[i].reshape(1, -1)
        sentence2 = ar2[j].reshape(1, -1)
       # sentence1 = ar1[i]
       # sentence2 = ar2[j]
        sim = cosine_similarity(sentence1, sentence2)
        sim = sim[0][0]
        sim_past_current.append(sim)

for i in range(len(past_smoker_set)):
    for j in range(len(past_smoker_set)):
        #calculate cosin sim between the two sentences
        sentence1 = ar1[i].reshape(1, -1)
        sentence2 = ar1[j].reshape(1, -1)
        #sentence1 = ar1[i]
       # sentence2 = past_smoker_set[j]
        sim = cosine_similarity(sentence1, sentence2)
        sim = sim[0][0]
        sim_past_past.append(sim)

for i in range(len(current_smoker_set)):
    for j in range(len(current_smoker_set)):
        #calculate cosin sim between the two sentences
        sentence1 = ar2[i].reshape(1, -1)
        sentence2 = ar2[j].reshape(1, -1)
        sim = cosine_similarity(sentence1, sentence2)
        sim = sim[0][0]
        sim_current_current.append(sim)


#averages
print(sum(sim_past_current) / len(sim_past_current))
print(sum(sim_past_past) / len(sim_past_past))
print(sum(sim_current_current) / len(sim_current_current))
"""
test_strings1 = ['Deze persoon rookt niet']

test_strings2 = ['Deze persoon rookt wel']

for i in range(len(test_strings1)):
    #calculate cosin sim between the two sentences
    sentence1 = ar1[i].reshape(1, -1)
    sentence2 = ar2[i].reshape(1, -1)
    print(cosine_similarity(sentence1, sentence2))
"""