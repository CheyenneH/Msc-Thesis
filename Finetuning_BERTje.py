import transformers
from transformers import AutoTokenizer, AutoModel, TFAutoModel, BertForPreTraining, BertForSequenceClassification, BertForTokenClassification
import torch
import pandas as pd
import random

"""
#testing
pipe = pipeline('fill-mask', model='GroNLP/bert-base-dutch-cased')
for res in pipe('Ik wou dat ik een [MASK] was.'):
    print(res['sequence'])
"""
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = BertForPreTraining.from_pretrained("GroNLP/bert-base-dutch-cased")  # PyTorch

#Data preperation
batch1 = 'Batch1.csv'
batch2 = 'Batch2.csv'
batch3 = 'Batch3.csv'

df_batch1 = pd.read_csv(batch1, delimiter=';', encoding='utf-8')
df_batch2 = pd.read_csv(batch2, delimiter=';', encoding='utf-8')
df_batch3 = pd.read_csv(batch3, delimiter=';', encoding='utf-8')

df_batch3 = df_batch3.drop(columns=['Unnamed: 4'])
df_batch2 = df_batch2.rename(columns={'TXT': 'txt'})

df_all_batches = pd.concat([df_batch1, df_batch2, df_batch3], axis=0)
#replace enters with dots, most sentences in the letters are seperated by enters
df_all_batches = df_all_batches.replace(r'\n','.', regex=True)

#only select the content of the letters
text = df_all_batches['txt'].tolist()
print(len(text))

#split on dot
text = [i.split('.') for i in text]
print(text[1])

#remove all elements which have less than 10 characters
#matrix = [[j for j in range(5)] for i in range(5)]
text = [[j for j in i if len(j)>= 10] for i in text]
bag = [item for sentence in text for item in sentence]
print("Number of sentences",len(bag))
text = text[:5000]
#Finetuning requires the following stuff:
# - NSP Next Sentence prediction. To prepare our data for NSP, we need to create a mix of non-random sentences
# (where the two sentences were originally together) — and random sentences.
# - Tokenization
# - MLM Masked Language Modeling/ Masking For MLM

#NSP-prep
#code to select 50/50 sentences that do follow each other and sentences that are just random
sentence_a = []
sentence_b = []
label = []
#1 is next sentence, 0 is not next sentence
for letter in text:
    num_sentences = len(letter)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        if random.random() >= 0.5:
            #classify as IsNextSetntence
            sentence_a.append(letter[start])
            sentence_b.append(letter[start + 1])
            label.append(1)
        else:
            #classify as NotNextSentence
            index = random.randint(0, len(bag) - 1)
            sentence_a.append(letter[start])
            sentence_b.append(bag[index])
            label.append(0)

for i in range(3):
    print(label[i])
    print(sentence_a[i] + '\n---')
    print(sentence_b[i] + '\n')

#Tokinization
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length')
print(inputs.keys())
print(inputs)


#Our NSP labels must be placed within a tensor called next_sentence_label.
#We create this easily by taking our label variable, and converting it into a
#torch.LongTensor — which must also be transposed using .T:
inputs['next_sentence_label'] = torch.LongTensor([label]).T
print(inputs.next_sentence_label[:10])


#Masking for MLM
#For MLM we need to clone our current input_ids tensor to create a MLM labels tensor
#then we move onto masking ~15% of tokens in the input_ids tensor.
inputs['labels'] = inputs.input_ids.detach().clone()
print(inputs.keys())
#Now that we that clone for our labels, we mask tokens in input_ids.
# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 1) * \
           (inputs.input_ids != 2) * (inputs.input_ids != 0)

#And now take the indices of each True value within each vector.
selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

#Then apply these indices to each row in input_ids, assigning each value at these indices a value of 103.
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103
print(inputs.keys())
print(inputs.input_ids)

#We create a PyTorch dataset from our data.
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


#Initialize our data using the OurDataset class.
dataset = OurDataset(inputs)
#And initialize the dataloader, which we'll be using to load our data into the model during training.
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

from torch.optim import AdamW

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

from tqdm import tqdm  # for our progress bar

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


model.save_pretrained("C:/Users/cheye/OneDrive - Universiteit Leiden/MASTER THESIS/Models")