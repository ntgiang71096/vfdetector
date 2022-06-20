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
from patch_entities import VulFixMinerDataset
from model import VulFixMinerClassifier, VulFixMinerFineTuneClassifier
import pandas as pd
from tqdm import tqdm
import utils
import config
import argparse
import vulfixminer_finetune
from transformers import RobertaTokenizer, RobertaModel
import csv 

# dataset_name = 'sap_patch_dataset.csv'
# EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_2'
# MODEL_PATH = 'model/patch_variant_2_finetune_1_epoch_best_model.sav'

dataset_name = None
FINETUNE_MODEL_PATH = None
MODEL_PATH = None
TRAIN_PROB_PATH = None
TEST_PROB_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(directory, 'model')


# retest with SAP dataset
NUMBER_OF_EPOCHS = 20
EARLY_STOPPING_ROUND = 5

TRAIN_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 256
HIDDEN_DIM = 768

NUMBER_OF_LABELS = 2


# model_path_prefix = model_folder_path + '/patch_variant_2_16112021_model_'


def predict_test_data(model, testing_generator, device, need_prob=False, need_feature_only=False, prob_path=None):
    y_pred = []
    y_test = []
    probs = []
    urls = []
    final_features = []
    with torch.no_grad():
        model.eval()
        for ids, url_batch, embedding_batch, label_batch in tqdm(testing_generator):
            embedding_batch, label_batch = embedding_batch.to(device), label_batch.to(device)

            outs = model(embedding_batch)
            if need_feature_only:
                final_features.extend(outs[1].tolist())
                outs = outs[0]

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

    if prob_path is not None:
        with open(prob_path, 'w') as file:
            writer = csv.writer(file)
            for i, prob in enumerate(probs):
                writer.writerow([urls[i], prob])


    if need_feature_only:
        return f1, urls, final_features

    if not need_prob:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, urls, probs


def train(model, learning_rate, number_of_epochs, training_generator, test_generator):
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
        for id_batch, url_batch, embedding_batch, label_batch in training_generator:
            embedding_batch, label_batch \
                = embedding_batch.to(device), label_batch.to(device)
            outs = model(embedding_batch)
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


    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), MODEL_PATH)
    else:
        torch.save(model.state_dict(), MODEL_PATH)

    return model


class CommitAggregator:
    def __init__(self, file_transformer):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.file_transformer = file_transformer

    def transform(self, diff_list):
        # cap at 20 diffs 
        diff_list = diff_list[:20]
        input_list, mask_list = [], []
        for diff in diff_list:
            added_code = vulfixminer_finetune.get_code_version(diff=diff, added_version=True)
            deleted_code = vulfixminer_finetune.get_code_version(diff=diff, added_version=False)

            code = added_code + self.tokenizer.sep_token + deleted_code
            input_ids, mask = vulfixminer_finetune.get_input_and_mask(self.tokenizer, [code])
            input_list.append(input_ids)
            mask_list.append(mask)

        input_list = torch.stack(input_list)
        mask_list = torch.stack(mask_list)
        input_list, mask_list = input_list.to(device), mask_list.to(device)
        embeddings = self.file_transformer(input_list, mask_list).last_hidden_state[:, 0, :]

        sum_ = torch.sum(embeddings, dim=0)
        mean_ = torch.div(sum_, len(diff_list))
        mean_ = mean_.detach()
        mean_ = mean_.cpu()

        return mean_


def do_train(args):
    global dataset_name, MODEL_PATH

    dataset_name = args.dataset_path
    FINETUNE_MODEL_PATH = args.finetune_model_path
    MODEL_PATH = args.model_path

    TRAIN_PROB_PATH = args.train_prob_path
    TEST_PROB_PATH = args.test_prob_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(MODEL_PATH))

    print("Loading finetuned file transformer...")
    finetune_model = VulFixMinerFineTuneClassifier()

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        finetune_model = nn.DataParallel(finetune_model)

    finetune_model.load_state_dict(torch.load(FINETUNE_MODEL_PATH))
    code_bert = finetune_model.module.code_bert
    code_bert.eval()
    code_bert.to(device)

    print("Finished loading")

    aggregator = CommitAggregator(code_bert)

    patch_data, label_data, url_data = vulfixminer_finetune.get_data(dataset_name)

    train_ids, test_ids = [], []

    index = 0

    id_to_embeddings, id_to_label, id_to_url = {}, {}, {}
    for i in tqdm(range(len(patch_data['train']))):
        label = label_data['train'][i]
        url = url_data['train'][i]
        embeddings = aggregator.transform(patch_data['train'][i])
        train_ids.append(index)
        id_to_embeddings[index] = embeddings
        id_to_label[index] = label
        id_to_url[index] = url
        # all_data.append(embeddings)
        # all_label.append(label)
        # all_url.append(url)
        index += 1

    for i in tqdm(range(len(patch_data['test']))):
        label = label_data['test'][i]
        url = url_data['test'][i]
        embeddings = aggregator.transform(patch_data['test'][i])
        test_ids.append(index)
        id_to_embeddings[index] = embeddings
        id_to_label[index] = label
        id_to_url[index] = url
        # all_data.append(embeddings)
        # all_label.append(label)
        # all_url.append(url)
        index += 1


    training_set = VulFixMinerDataset(train_ids, id_to_label, id_to_embeddings, id_to_url)
    test_set = VulFixMinerDataset(test_ids, id_to_label, id_to_embeddings, id_to_url)
    
    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    test_generator = DataLoader(test_set, **TEST_PARAMS)
   
    model = VulFixMinerClassifier()

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

    print("Writing result to file...")
    predict_test_data(model=model, testing_generator=training_generator, device=device, prob_path=TRAIN_PROB_PATH)
    predict_test_data(model=model, testing_generator=test_generator, device=device, prob_path=TEST_PROB_PATH)
    print("Finish writting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help='name of dataset')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='save train model to path')

    parser.add_argument('--finetune_model_path',
                        type=str,
                        required=True,
                        help='path to finetune file transfomer')

    parser.add_argument('--train_prob_path',
                        type=str,
                        required=True,
                        help='')

    parser.add_argument('--test_prob_path',
                        type=str,
                        required=True,
                        help='')
   
    args = parser.parse_args()


    do_train(args)
