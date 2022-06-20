from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from patch_entities import VulFixMinerFileDataset
from model import VulFixMinerFineTuneClassifier
from tqdm import tqdm
import pandas as pd
from utils import get_code_version
import config
import argparse

# dataset_name = 'sap_patch_dataset.csv'
# FINE_TUNED_MODEL_PATH = 'model/patch_variant_2_finetuned_model.sav'

dataset_name = None
FINE_TUNED_MODEL_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')

FINETUNE_EPOCH = 5

LIMIT_FILE_COUNT = 5

NUMBER_OF_EPOCHS = 5
TRAIN_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EARLY_STOPPING_ROUND = 5

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CODE_LENGTH = 256
HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.1
NUMBER_OF_LABELS = 2


def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'][0], inputs.data['attention_mask'][0]


def predict_test_data(model, testing_generator, device, need_prob=False):
    print("Testing...")
    y_pred = []
    y_test = []
    urls = []
    probs = []
    model.eval()
    with torch.no_grad():
        for id_batch, url_batch, input_batch, mask_batch, label_batch in tqdm(testing_generator):
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)

            outs = model(input_batch, mask_batch)
            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    if not need_prob:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, urls, probs


def train(model, learning_rate, number_of_epochs, training_generator, test_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
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
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []

        model.eval()

        print("Result on testing dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=test_generator,
                                                       device=device)
        
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        if epoch + 1 == FINETUNE_EPOCH:
            torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
            if not isinstance(model, nn.DataParallel):
                model.freeze_codebert()
            else:
                model.module.freeze_codebert()
    return model


def retrieve_patch_data(all_data, all_label, all_url):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Preparing tokenizer data...")

    id_to_label = {}
    id_to_url = {}
    id_to_input = {}
    id_to_mask = {}
    for i, diff in tqdm(enumerate(all_data)):
        added_code = get_code_version(diff=diff, added_version=True)
        deleted_code = get_code_version(diff=diff, added_version=False)

        code = added_code + tokenizer.sep_token + deleted_code

        input_ids, mask = get_input_and_mask(tokenizer, [code])
        id_to_input[i] = input_ids
        id_to_mask[i] = mask
        id_to_label[i] = all_label[i]
        id_to_url[i] = all_url[i]

    return id_to_input, id_to_mask, id_to_label, id_to_url


def get_sap_data(dataset_name):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
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

        url_to_diff[url].append(diff)
        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    return url_to_diff, url_to_partition, url_to_label, url_to_pl


def get_tensor_flow_data(dataset_name):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
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
        pl = "UNKNOWN"

        if url not in url_to_diff:
            url_to_diff[url] = []

        url_to_diff[url].append(diff)
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

    print(len(url_to_diff.keys()))
    # diff here is diff list
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


def do_train(args):
    global dataset_name, FINE_TUNED_MODEL_PATH

    dataset_name = args.dataset_path
    
    FINE_TUNED_MODEL_PATH = args.finetune_model_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(FINE_TUNED_MODEL_PATH))

    patch_data, label_data, url_data = get_data(dataset_name)

    train_ids, test_ids = [], []

    index = 0
    all_data, all_label, all_url = [], [], []

    for i in range(len(patch_data['train'])):
        label = label_data['train'][i]
        url = url_data['train'][i]
        for j in range(len(patch_data['train'][i])) :
            diff = patch_data['train'][i][j]
            train_ids.append(index)
            all_data.append(diff)
            all_label.append(label)
            all_url.append(url)
            index += 1

    for i in range(len(patch_data['test'])):
        label = label_data['test'][i]
        url = url_data['test'][i]
        for j in range(len(patch_data['test'][i])) :
            diff = patch_data['test'][i][j]
            test_ids.append(index)
            all_data.append(diff)
            all_label.append(label)
            all_url.append(url)
            index += 1


    print("Preparing commit patch data...")
    id_to_input, id_to_mask, id_to_label, id_to_url = retrieve_patch_data(all_data, all_label, all_url)
    print("Finish preparing commit patch data")

    training_set = VulFixMinerFileDataset(train_ids, id_to_label, id_to_url, id_to_input, id_to_mask)
    test_set = VulFixMinerFileDataset(test_ids, id_to_label, id_to_url, id_to_input, id_to_mask)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    test_generator = DataLoader(test_set, **TEST_PARAMS)

    model = VulFixMinerFineTuneClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          test_generator=test_generator)


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
