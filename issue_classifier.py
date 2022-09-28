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
from data_preprocessor import preprocess_single_record
from feature_options import ExperimentOption
from message_classifier import get_roberta_features, TextDataset, predict_test_data
import argparse
import config
from variant_8_finetune_separate import get_data
import json


directory = os.path.dirname(os.path.abspath(__file__))
github_issue_folder_path = os.path.join(directory, 'github_issues')

# dataset_name = os.path.join(directory, 'sub_enhanced_dataset_th_100.txt')
# MODEL_PATH = os.path.join(directory, 'model/issue_classifier.sav')

dataset_name = None
MODEL_PATH = None

# re-test with sap dataset epoch = 10
NUMBER_OF_EPOCHS = 20

learning_rate = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 8}

test_params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 8}

false_cases = []


def read_sap_issue(dataset_name='sub_enhanced_dataset_th_100.txt', need_urls=False):
    issues, labels, urls =  [], [], []
    url_to_issue = {}

    records = data_loader.load_records('sub_enhanced_dataset_th_100.txt')
    options = ExperimentOption()

    print("Preprocessing records...")

    records = [preprocess_single_record(record, options) for record in tqdm(records)]
    
    for record in records:
        url = record.repo + '/commit/' + record.commit_id
        url = url[len('https://github.com/') :]
        url_to_issue[url] = record.issue_info

    original_records = data_loader.load_records('full_dataset_with_all_features.txt')

    for record in original_records:
        url = record.repo + '/commit/' + record.commit_id
        url = url[len('https://github.com/') :]
        urls.append(url)
        labels.append(record.label)
        if url in url_to_issue:
            issues.append(url_to_issue[url])
        else:
            issues.append(' ')

    print("Finish preprocessing")

    if need_urls:
        return issues, labels, urls
    else: 
        return issues, labels


def read_issue():
    print("Reading Tensor Flow issues...")
    df = pd.read_csv('tf_issue_linking.csv')
    number_to_url = {}
    for item in df.values.tolist():
        number = item[1]
        url = item[0]
        if number not in number_to_url:
            number_to_url[number] = []
        number_to_url[number].append(url)

    url_to_issue = {}
    for file_name in os.listdir(github_issue_folder_path):
        if file_name.endswith('.json'):
            with open(github_issue_folder_path + '/' + file_name) as file:
                json_raw = file.read()
                issues = json.loads(json_raw)
                for issue in issues:
                    number = issue['number']
                    if number in number_to_url:
                        if issue['title'] is not None:
                            title = issue['title']
                        else:
                            title = ''

                        if issue['body'] is not None:
                            body = issue['body']
                        else:
                            body = ''
                        
                        for url in number_to_url[number]:
                            url_to_issue[url] = title + '\n' + body
    
    print("Finish reading")

    return url_to_issue


def read_tensor_flow_issue(dataset_name, need_url_data=False):
    patch_data, label_data, url_data = get_data(dataset_name)

    url_to_issue = read_issue()
    print(len(url_to_issue))
    text_train, text_test, label_train, label_test, url_train, url_test = [], [], [], [], [], []

    for i, url in enumerate(url_data['train']):
        text_train.append(url_to_issue[url])
        label_train.append(label_data['train'][i])
        url_train.append(url)

    for i, url in enumerate(url_data['test']):
        text_test.append(url_to_issue[url])
        label_test.append(label_data['test'][i])
        url_test.append(url)

    if not need_url_data:
        return text_train, text_test, label_train, label_test
    else:
        return text_train, text_test, label_train, label_test, url_train, url_test


def do_train(args):

    global dataset_name, MODEL_PATH

    dataset_name = args.dataset_path
    
    MODEL_PATH = args.model_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(MODEL_PATH))

    if dataset_name == config.SAP_DATASET_NAME:
        texts, labels = read_sap_issue(dataset_name)
        text_train, text_test, label_train, label_test = train_test_split(texts, labels, test_size=0.20, random_state=109)
    else:
        text_train, text_test, label_train, label_test = read_tensor_flow_issue(dataset_name)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Converting using pretrained...")

    train_features = get_roberta_features(tokenizer, text_train, length=256)
    test_features = get_roberta_features(tokenizer, text_test, length=256)

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

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

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
