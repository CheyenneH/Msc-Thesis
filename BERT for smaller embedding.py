from transformers import AutoTokenizer, AutoModel, TFAutoModel, BertForPreTraining, BertForSequenceClassification, BertForTokenClassification
import torch
import pandas as pd
import numpy as np

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

for index, row in df.iterrows():
    string = row['txt'].lower()
    # find word "roken"
    smoking_index = string.find(" roken")
    if smoking_index == -1:
        mylist.append(string[:40])
        #print("smoking  not found but string is lengeth", len(string[:40]))
        continue
    #get 40 positions before and 50 positions after #this is something I can experiment with
    start = smoking_index - 20
    end = smoking_index + 20
    new_string = string[start:end]
    #print(len(new_string))
    mylist.append(new_string)

df['new_string'] = mylist
new_df = df['new_string']
new_df = new_df.replace({'.':''}, regex=True)
text = new_df
"""

test_strings1 = ['roken gestopt sinds 32 jaar, diabetes mellitus, bmi 26',
                 'roken gestopt sinds 32 jaar, diabetes mellitus, bmi 26',
                 'rookt niet meer',]

test_strings2 = ['verslag gesprek met patiënt: gaat beter. minder dyspnoisch, waarschijnlijk toch door stoppen met roken.',
                 'intoxicaties: roken- alk niet meer, daarvoor weinig',
                 'roken--' ]

"""
test_strings1= ['Deze persoon rookt niet']
                
test_strings2= ['Deze persoon rookt wel']
                
                

#apply embedding
embeddings1 = []
# ---Add markers for each sentence
for letter_num in range(len(test_strings1)):
    print(letter_num, "out of ", len(test_strings1))

    marked_text = "[CLS] " + test_strings1[letter_num] + " [SEP]"
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
for letter_num in range(len(test_strings2)):
    print(letter_num, "out of ", len(test_strings2))

    marked_text = "[CLS] " + test_strings2[letter_num] + " [SEP]"
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

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(test_strings1)):
    #calculate cosin sim between the two sentences
    sentence1 = ar1[i].reshape(1, -1)
    sentence2 = ar2[i].reshape(1, -1)
    print(cosine_similarity(sentence1, sentence2))
#np.savetxt("small_embedding.csv", ar, fmt='%s', delimiter=",")
#final_labels["label"].to_csv("labels.csv", index=False)
