import torch
from torch import nn as nn
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from patch_entities import VariantEightFineTuneOnlyDataset
from model import VariantEightFineTuneOnlyClassifier
import pandas as pd
from tqdm import tqdm
import utils
from transformers import RobertaTokenizer
import argparse
import config

# dataset_name = 'sap_patch_dataset.csv'
# FINE_TUNED_MODEL_PATH = 'model/patch_variant_8_finetuned_model.sav'

dataset_name = None
FINE_TUNED_MODEL_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(directory, 'model')

FINETUNE_EPOCH = 5

NUMBER_OF_EPOCHS = 5
EARLY_STOPPING_ROUND = 5

TRAIN_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 64
HIDDEN_DIM = 768

NUMBER_OF_LABELS = 2


def train(model, learning_rate, number_of_epochs, training_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for id_batch, url_batch, input_batch, mask_batch, label_batch in tqdm(training_generator):
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)
            outs = model(input_batch, mask_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, average loss {}"
                      .format(current_batch, np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))

        torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)


        # if epoch + 1 == FINETUNE_EPOCH:
        #     torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
        #     if not isinstance(model, nn.DataParallel):
        #         model.freeze_codebert()
        #     else:
        #         model.module.freeze_codebert()

    return model


def get_sap_data(dataset_name):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        diff = item[3]
        label = item[4]
        pl = item[5]

        if url not in url_to_diff:
            url_to_diff[url] = []

        removed_code = utils.get_code_version(diff, False)
        added_code = utils.get_code_version(diff, True)

        new_removed_code_list = utils.get_line_from_code(tokenizer.sep_token, removed_code)
        new_added_code_list = utils.get_line_from_code(tokenizer.sep_token, added_code)

        url_to_diff[url].extend(new_removed_code_list)
        url_to_diff[url].extend(new_added_code_list)

        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    return url_to_diff, url_to_partition, url_to_label, url_to_pl


def get_tensor_flow_data(dataset_name):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    df = df[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[6]
        diff = item[4]

        if pd.isnull(diff):   
            continue

        label = item[5]
        pl = 'UNKNOWN'

        if url not in url_to_diff:
            url_to_diff[url] = []

        removed_code = utils.get_code_version(diff, False)
        added_code = utils.get_code_version(diff, True)

        new_removed_code_list = utils.get_line_from_code(tokenizer.sep_token, removed_code)
        new_added_code_list = utils.get_line_from_code(tokenizer.sep_token, added_code)

        url_to_diff[url].extend(new_removed_code_list)
        url_to_diff[url].extend(new_added_code_list)

        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    return url_to_diff, url_to_partition, url_to_label, url_to_pl


def get_data(dataset_name):
    if dataset_name == config.SAP_DATASET_NAME:
        url_to_diff, url_to_partition, url_to_label, url_to_pl = get_sap_data(dataset_name)
    else:
        url_to_diff, url_to_partition, url_to_label, url_to_pl = get_tensor_flow_data(dataset_name)

    patch_train, patch_test = [], []
    label_train, label_test = [], []
    url_train, url_test = [], []

    for key in url_to_diff.keys():
        url = key
        diff = url_to_diff[key]
        label = url_to_label[key]
        partition = url_to_partition[key]
        pl = url_to_pl[key]
        if partition == 'train':
            patch_train.append(diff)
            label_train.append(label)
            url_train.append(url)
        elif partition == 'test':
            patch_test.append(diff)
            label_test.append(label)
            url_test.append(url)

    print("Finish reading dataset")
    patch_data = {'train': patch_train, 'test': patch_test}

    label_data = {'train': label_train, 'test': label_test}

    url_data = {'train': url_train, 'test': url_test}

    return patch_data, label_data, url_data


def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'], inputs.data['attention_mask']


def retrieve_patch_data(all_data, all_label, all_url):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Preparing tokenizer data...")

    id_to_label = {}
    id_to_url = {}
    id_to_input = {}
    id_to_mask = {}
    index = 0
    for i, line_list in tqdm(enumerate(all_data)):
        code_list = []

        for count, line in enumerate(line_list):
            code = tokenizer.sep_token + line
            code_list.append(code)
        
        if len(code_list) == 0:
            code_list = [tokenizer.sep_token]
            
        input_ids_list, mask_list = get_input_and_mask(tokenizer, code_list)
        for j in range(len(input_ids_list)):
            id_to_input[index] = input_ids_list[j]
            id_to_mask[index] = mask_list[j]
            id_to_label[index] = all_label[i]
            id_to_url[index] = all_url[i]
            index += 1

    return id_to_input, id_to_mask, id_to_label, id_to_url


def do_train(args):
    global dataset_name, FINE_TUNED_MODEL_PATH

    dataset_name = args.dataset_path
    
    FINE_TUNED_MODEL_PATH = args.finetune_model_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(FINE_TUNED_MODEL_PATH))

    patch_data, label_data, url_data = get_data(dataset_name)

    train_ids, test_ids = [], []

    index = 0
    for i, line_list in enumerate((patch_data['train'])):
        for j in range(len(line_list)):
            train_ids.append(index)
            index += 1

    all_data = patch_data['train']
    all_label = label_data['train']
    all_url = url_data['train']

    print("Preparing commit patch data...")
    id_to_input, id_to_mask, id_to_label, id_to_url = retrieve_patch_data(all_data, all_label, all_url)
    print("Finish preparing commit patch data")

    training_set = VariantEightFineTuneOnlyDataset(train_ids, id_to_label, id_to_url, id_to_input, id_to_mask)
    training_generator = DataLoader(training_set, **TRAIN_PARAMS)

    model = VariantEightFineTuneOnlyClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help='name of dataset')
    parser.add_argument('--finetune_model_path',
                        type=str,
                        required=True,
                        help='select path to save model')

    args = parser.parse_args()

    do_train(args)
