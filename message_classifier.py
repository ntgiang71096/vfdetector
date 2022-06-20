from socket import MsgFlag
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from torch import optim as optim
from sklearn import metrics
import numpy as np
import random
import math
import csv
from transformers import AdamW
from transformers import get_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import data_loader
import argparse
import config
import utils

directory = os.path.dirname(os.path.abspath(__file__))

# dataset_name = 'full_dataset_with_all_features.txt')
# MODEL_PATH = os.path.join(directory, 'model/message_classifier.sav')

dataset_name = None
MODEL_PATH = None

NUMBER_OF_EPOCHS = 20
learning_rate = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 8}

test_params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 8}


class TextDataset(Dataset):
    def __init__(self, list_IDs, labels, id2input, id2mask):
        self.labels = labels
        self.list_IDs = list_IDs
        self.id2input = id2input
        self.id2mask = id2mask

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        input_ids = self.id2input[ID]
        mask = self.id2mask[ID]
        y = self.labels[ID]
        return int(ID), input_ids, mask, y


def predict_test_data(model, testing_generator, device, writing_false_case=False, need_probs=False):
    y_pred = []
    y_test = []
    y_probs = []
    failed_predictions = []
    with torch.no_grad():
        model.eval()
        for id_batch, input_id_batch, mask_batch, label_batch in testing_generator:
            input_id_batch, mask_batch, label_batch \
                = input_id_batch.to(device), mask_batch.to(device), label_batch.to(device)

            outs = model(input_id_batch, mask_batch)

            y_pred.extend(torch.argmax(outs.logits, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            
            y_probs.extend((F.softmax(outs.logits, dim=1))[:, 1].tolist())

        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

    if need_probs:
        return precision, recall, f1, y_probs
    else:
        return precision, recall, f1


def read_message(file_path, is_pos):
    message_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_ALL, delimiter=';')
        for row in reader:
            if is_pos:
                message_list.append(row[3])
            else:
                message_list.append(row[0])
    return message_list

def read_sap_dataset(need_urls=False):
    records = data_loader.load_records(dataset_name)
    messages = []
    labels = []
    urls = []
    for record in records:
        messages.append(record.commit_message)
        labels.append(record.label)
        url = record.repo + '/commit/' + record.commit_id
        url = url[len('https://github.com/') :]
        urls.append(url)

    if need_urls:
        return messages, labels, urls
    else:
        return messages, labels


def get_roberta_features(tokenizer, messages, length=128):
    features = []
    for message in tqdm(messages):
        inputs = tokenizer(message, padding='max_length', max_length=length, truncation=True, return_tensors="pt")
        features.append((inputs.data['input_ids'][0], inputs.data['attention_mask'][0]))
    return features


def read_tensor_flow_dataset(dataset_name, need_url_data=False):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)

    df = df[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition']]

    patch_data, label_data, url_data = utils.get_data(dataset_name)

    items = df.to_numpy().tolist()

    url_to_msg, url_to_partition, url_to_label = {}, {}, {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[6]
        message = item[2]

        if pd.isnull(message):   
            message = ' '

        label = item[5]
        pl = 'UNKNOWN'
        
        url_to_msg[url] = message
        url_to_label[url] = label
        url_to_partition[url] = partition

    message_train, message_test, label_train, label_test, url_train, url_test = [], [], [], [], [], []

    for i, url in enumerate(url_data['train']):
        message_train.append(url_to_msg[url])
        label_train.append(label_data['train'][i])
        url_train.append(url)

    for i, url in enumerate(url_data['test']):
        message_test.append(url_to_msg[url])
        label_test.append(label_data['test'][i])
        url_test.append(url)

    if not need_url_data:
        return message_train, message_test, label_train, label_test
    else:
        return message_train, message_test, label_train, label_test, url_train, url_test


def do_train(args):

    global dataset_name, MODEL_PATH

    dataset_name = args.dataset_path
    
    MODEL_PATH = args.model_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(MODEL_PATH))

    if dataset_name == config.SAP_DATASET_NAME:
        messages, labels = read_sap_dataset()
        message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=0.20, random_state=109)
    else:
        message_train, message_test, label_train, label_test = read_tensor_flow_dataset(dataset_name)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Converting using pretrained...")

    train_features = get_roberta_features(tokenizer, message_train)
    test_features = get_roberta_features(tokenizer, message_test)

    print("Finish preparing!")
    train_partition = []
    test_partition = []
    partition = {}
    labels = {}
    id2input = {}
    id2mask = {}

    for i in range(len(train_features)):
        id = i
        input_id = train_features[i][0]
        attention_mask = train_features[i][1]
        label = label_train[i]
        train_partition.append(id)
        labels[id] = label
        id2input[id] = input_id
        id2mask[id] = attention_mask

    partition['train'] = train_partition

    for i in range(len(test_features)):
        id = len(train_features) + i           # next index
        input_id = test_features[i][0]
        attention_mask = test_features[i][1]
        label = label_test[i]
        test_partition.append(id)
        labels[id] = label
        id2input[id] = input_id
        id2mask[id] = attention_mask

    partition['test'] = test_partition

    training_set = TextDataset(partition['train'], labels, id2input, id2mask)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    testing_set = TextDataset(partition['test'], labels, id2input, id2mask)
    testing_generator = torch.utils.data.DataLoader(testing_set, **params)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.to(device)

    model.train()
    for epoch in range(NUMBER_OF_EPOCHS):
        total_loss = 0
        current_batch = 0
        for id_batch, input_id_batch, mask_batch, label_batch in training_generator:
            # print("epoch {} current_batch {}".format(epoch, current_batch))
            current_batch += 1
            input_id_batch, mask_batch, label_batch \
                = input_id_batch.to(device), mask_batch.to(device), label_batch.to(device)
            outs = model(input_ids=input_id_batch, attention_mask=mask_batch, labels=label_batch)
            loss = outs.loss
            # logits = outs.logits
            # loss = loss_function(F.softmax(logits, dim=1), label_batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            total_loss += loss.detach().item()

        print("epoch {}, learning rate {}, total loss {}".format(epoch, lr_scheduler.get_last_lr(), total_loss))

        precision, recall, f1 = predict_test_data(model=model,
                                                  testing_generator=testing_generator,
                                                  device=device)
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        # print("AUC Java: {}".format(auc))

    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help='name of dataset')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='select path to save model')

    args = parser.parse_args()

    do_train(args)