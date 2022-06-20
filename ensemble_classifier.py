import os
import json
import utils
from torch.utils.data import DataLoader
from patch_entities import EnsembleDataset
from model import EnsembleModel
import torch
from torch import cuda
from torch import nn as nn
from transformers import AdamW
from transformers import get_scheduler
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import csv
import argparse
import configparser

dataset_name = None
MODEL_PATH = None
TRAIN_PROBS_PATH = None
TEST_PROBS_PATH = None

TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

directory = os.path.dirname(os.path.abspath(__file__))
TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

# re-test with sap dataset
NUMBER_OF_EPOCHS = 20

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def write_prob_to_file(file_path, urls, probs):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(urls):
            writer.writerow([url, probs[i]])


def read_features_from_file(file_path):
    file_path = os.path.join(directory, file_path)
    with open(file_path, 'r') as reader:
        data = json.loads(reader.read())

    return data


def read_feature_list(file_path_list, reshape=False, need_list=False):
    url_to_feature = {}
    for file_path in file_path_list:
        data = read_features_from_file(file_path)
        for url, feature in data.items():
            if url not in url_to_feature:
                url_to_feature[url] = []
            url_to_feature[url].append(feature)

    if not reshape:
        return url_to_feature
    else:
        url_to_combined = {}
        if reshape:
            for url in url_to_feature.keys():
                features = url_to_feature[url]
                combine = []
                for feature in features:
                    combine.extend(feature)
                if not need_list:
                    combine = torch.FloatTensor(combine)
                url_to_combined[url] = combine

        return url_to_combined


def predict_test_data(model, testing_generator, device, need_prob=False):
    y_pred = []
    y_test = []
    probs = []
    urls = []
    with torch.no_grad():
        model.eval()
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(testing_generator):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)

            label_batch = label_batch.to(device)

            outs = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)

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
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(training_generator):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)

            label_batch = label_batch.to(device)

            outs = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)
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
        precision, recall, f1, auc, urls, probs = predict_test_data(model=model,
                                                       testing_generator=test_generator,
                                                       device=device, need_prob=True)

        if epoch == number_of_epochs - 1:
            write_prob_to_file(TEST_PROBS_PATH, urls, probs)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        if epoch == number_of_epochs - 1:
            torch.save(model.state_dict(), MODEL_PATH)

    return model


def do_train(args):
    global dataset_name, MODEL_PATH, TRAIN_PROBS_PATH, TEST_PROBS_PATH

    config_file_name = args.config_file
    config_parser = configparser.RawConfigParser()
    config_parser.read(config_file_name) 
    config_dict = dict(config_parser.items('DATASET_CONFIG'))

    dataset_name = config_dict['dataset_name']
    MODEL_PATH = config_dict['ensemble_model_path']
    TRAIN_PROBS_PATH = config_dict['ensemble_train_prob_path']
    TEST_PROBS_PATH = config_dict['ensemble_test_prob_path']


    variant_to_drop = []
    if args.ablation_study == True:
        for variant in args.variant_to_drop:
            variant_to_drop.append(int(variant))

    train_feature_path = [
        config_dict['variant_one_train_feature_path'],
        config_dict['variant_two_train_feature_path'],
        config_dict['variant_three_train_feature_path'],
        config_dict['variant_five_train_feature_path'],
        config_dict['variant_six_train_feature_path'],
        config_dict['variant_seven_train_feature_path'],
        config_dict['variant_eight_train_feature_path']
    ]

    test_feature_path = [
        config_dict['variant_one_test_feature_path'],
        config_dict['variant_two_test_feature_path'],
        config_dict['variant_three_test_feature_path'],
        config_dict['variant_five_test_feature_path'],
        config_dict['variant_six_test_feature_path'],
        config_dict['variant_seven_test_feature_path'],
        config_dict['variant_eight_test_feature_path']
    ]

    print("Reading data...")
    url_to_features = {}
    print("Reading train data")
    url_to_features.update(read_feature_list(train_feature_path))
    print("Reading test  data")
    url_to_features.update(read_feature_list(test_feature_path))

    print("Finish reading")
    url_data, label_data = utils.get_data(dataset_name)

    feature_data = {}
    feature_data['train'] = []
    feature_data['test'] = []

    for url in url_data['train']:
        feature_data['train'].append(url_to_features[url])

    for url in url_data['test']:
        feature_data['test'].append(url_to_features[url])

    train_ids, test_ids = [], []
    index = 0
    id_to_url = {}
    id_to_label = {}
    id_to_feature = {}

    for i, url in enumerate(url_data['train']):
        train_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['train'][i]
        id_to_feature[index] = feature_data['train'][i]
        index += 1

    for i, url in enumerate(url_data['test']):
        test_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test'][i]
        id_to_feature[index] = feature_data['test'][i]
        index += 1


    training_set = EnsembleDataset(train_ids, id_to_label, id_to_url, id_to_feature)
    test_set = EnsembleDataset(test_ids, id_to_label, id_to_url, id_to_feature)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    test_generator = DataLoader(test_set, **TEST_PARAMS)

    model = EnsembleModel(args.ablation_study, variant_to_drop)

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

    print("Result on testing dataset...")
    precision, recall, f1, auc, urls, probs = predict_test_data(model=model,
                                                    testing_generator=training_generator,
                                                    device=device, need_prob=True)

    write_prob_to_file(TRAIN_PROBS_PATH, urls, probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Classifier')
    parser.add_argument('--config_file',
                        type=str,
                        required=True,
                        help='Config file based on dataset')
    parser.add_argument('--ablation_study',
                        type=bool,
                        default=False,
                        help='Do ablation study or not')
    parser.add_argument('-v',
                        '--variant_to_drop',
                        action='append',
                        required=False,
                        help='Select index of variant to drop, 1, 2, 3, 5, 6, 7, 8')
    # parser.add_argument('--model_path',
    #                     type=str,
    #                     help='IMPORTANT select path to save model')
    # parser.add_argument('--train_probs_path',
    #                    type=str,
    #                    help='path to save predicted probabilities on training dataset')
    # parser.add_argument('--test_probs_path',
    #                    type=str,
    #                    help='path to save predicted probabilities on testing dataset')

    args = parser.parse_args()
    do_train(args)