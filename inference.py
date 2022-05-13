import os
import json
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

BATCH_SIZE = 32
max_length = 384
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference...')

def get_data(data_path):
    with open(data_path, 'r') as json_file:
        json_list = list(json_file)
        data = [json.loads(json_str) for json_str in json_list]
        print('Data size:', len(data))
        return data

def preprocess(articles):
    encode_articles = {}

    maintext = [article["maintext"] for article in articles]
    
    # Tokenize
    encode_articles = tokenizer(maintext, 
                                max_length=max_length,
                                truncation=True, 
                                padding=True)
    return encode_articles

class Summary(Dataset):
    def __init__(self, encoded_dataset):
        self.encode_token = encoded_dataset["input_ids"]
        
    def __getitem__(self, index):
        return torch.tensor(self.encode_token[index])

    def __len__(self):
        return len(self.encode_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to the data file")
    parser.add_argument("output_path", help="path to the output predictions")
    args = parser.parse_args()

    ## load data
    print('Data loading...')
    data = get_data(args.data_path)
    encoded_dataset = preprocess(data)
    valid = Summary(encoded_dataset)
    validloader = DataLoader(dataset=valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ## load model 
    print('Model loading...')
    config = AutoConfig.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained('./checkpoint-20000.pt', config=config)
    model.to(device)

    print('Predicting...')
    model.eval()
    eval_predictions = []
    with torch.no_grad():
        for input_ids in validloader:
            input_ids = input_ids.to(device)
            outputs = model.generate(input_ids, num_beams=5, max_length=64)
            eval_predictions += outputs

    decode_pred = [tokenizer.decode(p, skip_special_tokens=True) for p in eval_predictions]
    ids = [d['id'] for d in data]

    with open(args.output_path, 'w') as f:
        for t, i in zip(decode_pred, ids):
            f.write("{"+f'"title": "{t}", "id": "{i}"'+"}\n")
