import torch
from torch import nn as nn
import os
import csv
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import cuda
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification
from patch_entities import VariantOneDataset, VariantTwoDataset, VariantFiveDataset, VariantSixDataset, VariantThreeDataset, \
    VariantSevenDataset, VariantEightDataset
from model import VariantOneClassifier, VariantTwoClassifier, VariantFiveClassifier, VariantSixClassifier, \
    VariantThreeClassifier, VariantSevenClassifier, VariantEightClassifier
import utils
import variant_8_finetune_separate
from sklearn import metrics
from statistics import mean
from sklearn.linear_model import LogisticRegression
import csv
import json
import message_classifier
import issue_classifier
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import argparse
import configparser
import config
import pickle

directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

# dataset_name = 'sap_patch_dataset.csv'
# VARIANT_ONE_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_1'
# VARIANT_TWO_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_2'
# VARIANT_THREE_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_3'
# VARIANT_FIVE_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_5'
# VARIANT_SIX_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_6'
# VARIANT_SEVEN_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_7'
# VARIANT_EIGHT_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_8'
# MESSAGE_MODEL_PATH = 'model/message_classifier.sav'
# ISSUE_MODEL_PATH = 'model/issue_classifier.sav'
# VARIANT_ONE_MODEL_PATH = 'model/patch_variant_1_finetune_1_epoch_best_model.sav'
# VARIANT_TWO_MODEL_PATH = 'model/patch_variant_2_finetune_1_epoch_best_model.sav'
# VARIANT_THREE_MODEL_PATH = 'model/patch_variant_3_finetune_1_epoch_best_model.sav'
# VARIANT_FIVE_MODEL_PATH = 'model/patch_variant_5_finetune_1_epoch_best_model.sav'
# VARIANT_SIX_MODEL_PATH = 'model/patch_variant_6_finetune_1_epoch_best_model.sav'
# VARIANT_SEVEN_MODEL_PATH = 'model/patch_variant_7_finetune_1_epoch_best_model.sav'
# VARIANT_EIGHT_MODEL_PATH = 'model/patch_variant_8_finetune_1_epoch_best_model.sav'

dataset_name = None
VARIANT_ONE_EMBEDDINGS_DIRECTORY = None
VARIANT_TWO_EMBEDDINGS_DIRECTORY = None
VARIANT_THREE_EMBEDDINGS_DIRECTORY = None
VARIANT_FIVE_EMBEDDINGS_DIRECTORY = None
VARIANT_SIX_EMBEDDINGS_DIRECTORY = None
VARIANT_SEVEN_EMBEDDINGS_DIRECTORY = None
VARIANT_EIGHT_EMBEDDINGS_DIRECTORY = None
MESSAGE_MODEL_PATH = None
ISSUE_MODEL_PATH = None
VARIANT_ONE_MODEL_PATH = None
VARIANT_TWO_MODEL_PATH = None
VARIANT_THREE_MODEL_PATH = None
VARIANT_FIVE_MODEL_PATH = None
VARIANT_SIX_MODEL_PATH = None
VARIANT_SEVEN_MODEL_PATH = None
VARIANT_EIGHT_MODEL_PATH = None

TEST_BATCH_SIZE = 128

TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
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


def write_feature_to_file(file_path, urls, features):
    file_path = os.path.join(directory, file_path)
    data = {}
    for i, url in enumerate(urls):
        data[url] = features[i]

    json.dump(data, open(file_path, 'w'))


def infer_variant_1(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantOneClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    model.module.load_state_dict(torch.load(VARIANT_ONE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantOneDataset(ids, id_to_label, id_to_url, VARIANT_ONE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        f1, urls, features = variant_1.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)

    else:
        precision, recall, f1, auc, urls, probs = variant_1.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_2(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantTwoClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.module.load_state_dict(torch.load(VARIANT_TWO_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantTwoDataset(ids, id_to_label, id_to_url, VARIANT_TWO_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        f1, urls, features = variant_2.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_2.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_3(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantThreeClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.module.load_state_dict(torch.load(VARIANT_THREE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantThreeDataset(ids, id_to_label, id_to_url, VARIANT_THREE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_3.custom_collate)

    if need_feature_only:
        f1, urls, features = variant_3.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_3.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_5(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantFiveClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.module.load_state_dict(torch.load(VARIANT_FIVE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantFiveDataset(ids, id_to_label, id_to_url, VARIANT_FIVE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        f1, urls, features = variant_5.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_5.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_6(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantSixClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.module.load_state_dict(torch.load(VARIANT_SIX_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSixDataset(ids, id_to_label, id_to_url, VARIANT_SIX_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        f1, urls, features = variant_6.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_6.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_7(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantSevenClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.module.load_state_dict(torch.load(VARIANT_SEVEN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSevenDataset(ids, id_to_label, id_to_url, VARIANT_SEVEN_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_7.custom_collate)

    if need_feature_only:
        f1, urls, features = variant_7.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_7.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_8(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantEightClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.module.load_state_dict(torch.load(VARIANT_EIGHT_MODEL_PATH, map_location={'cuda:0': 'cuda:1'}))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantEightDataset(ids, id_to_label, id_to_url, VARIANT_EIGHT_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_8.custom_collate)

    if need_feature_only:
        f1, urls, features = variant_8.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("F1: {}".format(f1))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_8.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def get_dataset_info(partition):
    print("Dataset name: {}".format(dataset_name))
    url_data, label_data = utils.get_data(dataset_name)
    ids = []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data[partition]):
        ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data[partition][i]
        index += 1

    return ids, id_to_label, id_to_url


def read_pred_prob(file_path):
    df = pd.read_csv(file_path, header=None)
    url_to_prob = {}

    for url, prob in df.values.tolist():
        url_to_prob[url] = prob

    return url_to_prob


def get_auc_max_ensemble():
    print("Reading result...")
    variant_1_result = read_pred_prob('probs/prob_variant_1_finetune_1_epoch_test_python.txt')
    variant_2_result = read_pred_prob('probs/prob_variant_2_finetune_1_epoch_test_python.txt')
    variant_3_result = read_pred_prob('probs/prob_variant_3_finetune_1_epoch_test_python.txt')
    variant_5_result = read_pred_prob('probs/prob_variant_5_finetune_1_epoch_test_python.txt')
    variant_6_result = read_pred_prob('probs/prob_variant_6_finetune_1_epoch_test_python.txt')
    variant_7_result = read_pred_prob('probs/prob_variant_7_finetune_1_epoch_test_python.txt')
    variant_8_result = read_pred_prob('probs/prob_variant_8_finetune_1_epoch_test_python.txt')

    print("Finish reading result")

    url_to_max_prob = {}

    for url, prob_1 in variant_1_result.items():
        prob_2 = variant_2_result[url]
        prob_3 = variant_3_result[url]
        prob_5 = variant_5_result[url]
        prob_6 = variant_6_result[url]
        prob_7 = variant_7_result[url]
        prob_8 = variant_8_result[url]
        # url_to_max_prob[url] = mean([prob_1, prob_2, prob_3, prob_8])
        url_to_max_prob[url] = mean([prob_1, prob_2, prob_3, prob_5, prob_6, prob_7, prob_8])

    url_data, label_data = utils.get_data(dataset_name)
    url_test = url_data['test_python']
    label_test = label_data['test_python']

    y_score = []
    y_true = []
    for i, url in enumerate(url_test):
        y_true.append(label_test[i])
        y_score.append(url_to_max_prob[url])

    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)

    print("AUC: {}".format(auc))

    with open('probs/mean_prob_python.txt', 'w') as file:
        writer = csv.writer(file)
        for url, prob in url_to_max_prob.items():
            writer.writerow([url, prob])


def get_data_ensemble_model(prob_list, label_list):
    clf = LogisticRegression(random_state=109).fit(prob_list, label_list)

    return clf


def get_variant_result(variant_result_path):
    result = read_pred_prob(variant_result_path)

    return result


def get_prob(result_list, url):
    return [result[url] for result in result_list]


def get_partition_prob_list(result_path_list, partition):
    result_list = []
    for result_path in result_path_list:
        variant_result = get_variant_result(result_path)
        result_list.append(variant_result)

    url_data, label_data = utils.get_data(dataset_name)

    prob_list, label_list = [], []

    for i, url in enumerate(url_data[partition]):
        prob_list.append(get_prob(result_list, url))
        label_list.append(label_data[partition][i])

    return prob_list, label_list, url_data[partition]

def get_combined_ensemble_model():
    train_result_path_list = [
        'probs/variant_1_prob_train.txt',
        'probs/variant_2_prob_train.txt',
        'probs/variant_3_prob_train.txt',
        'probs/variant_5_prob_train.txt',
        'probs/variant_6_prob_train.txt',
        'probs/variant_7_prob_train.txt',
        'probs/variant_7_prob_train.txt'
    ]

    test_result_path_list = [ 'probs/variant_1_prob_test.txt',
        'probs/variant_2_prob_test.txt',
        'probs/variant_3_prob_test.txt',
        'probs/variant_5_prob_test.txt',
        'probs/variant_6_prob_test.txt',
        'probs/variant_7_prob_test.txt',
        'probs/variant_7_prob_test.txt']


    train_prob_list, train_label_list, train_url_list = get_partition_prob_list(train_result_path_list, 'train')
    test_prob_list, test_label_list, test_url_list = get_partition_prob_list(test_result_path_list, 'test')

    train_ensemble_model = get_data_ensemble_model(train_prob_list, train_label_list)
    print("Training ensemble model...")
    print("Finish training")

    print("Calculate on test dataset...")
    y_probs = train_ensemble_model.predict_proba(test_prob_list)[:, 1]
    y_pred = train_ensemble_model.predict(test_prob_list)
    f1 = metrics.f1_score(y_true=test_label_list, y_pred=y_pred)
    print("F1 of ensemble model: {}".format(f1))

    with open('probs/patch_ensemble_prob_test.txt', 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(test_url_list):
            writer.writerow([url, y_probs[i]])


    # predict on train dataset for fusion with message and issue classifier later

    y_train_probs = train_ensemble_model.predict_proba(train_prob_list)[:, 1]
    with open('probs/patch_ensemble_prob_train.txt', 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(train_url_list):
            writer.writerow([url, y_train_probs[i]])


def infer_all_variant(config_dict):
    print("Inferring variant 1...")
    infer_variant_1('train', config_dict['VARIANT_ONE_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_1('test', config_dict['VARIANT_ONE_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_1('train', config_dict['VARIANT_ONE_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_1('test', config_dict['VARIANT_ONE_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)
    
    print("Inferring variant 2...")
    infer_variant_2('train', config_dict['VARIANT_TWO_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_2('test', config_dict['VARIANT_TWO_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_2('train', config_dict['VARIANT_TWO_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_2('test', config_dict['VARIANT_TWO_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)

    print("Inferring variant 3...")
    infer_variant_3('train', config_dict['VARIANT_THREE_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_3('test', config_dict['VARIANT_THREE_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_3('train', config_dict['VARIANT_THREE_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_3('test', config_dict['VARIANT_THREE_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)

    print("Inferring variant 5...")
    infer_variant_5('train', config_dict['VARIANT_FIVE_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_5('test', config_dict['VARIANT_FIVE_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_5('train', config_dict['VARIANT_FIVE_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_5('test', config_dict['VARIANT_FIVE_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)

    print("Inferring variant 6...")
    infer_variant_6('train', config_dict['VARIANT_SIX_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_6('test', config_dict['VARIANT_SIX_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_6('train', config_dict['VARIANT_SIX_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_6('test', config_dict['VARIANT_SIX_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)

    print("Inferring variant 7...")
    infer_variant_7('train', config_dict['VARIANT_SEVEN_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_7('test', config_dict['VARIANT_SEVEN_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_7('train', config_dict['VARIANT_SEVEN_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_7('test', config_dict['VARIANT_SEVEN_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)

    print("Inferring variant 8...")
    infer_variant_8('train', config_dict['VARIANT_EIGHT_TRAIN_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_8('test', config_dict['VARIANT_EIGHT_TEST_FEATURE_PATH'.lower()], need_feature_only=True)
    infer_variant_8('train', config_dict['VARIANT_EIGHT_TRAIN_PROB_PATH'.lower()], need_feature_only=False)
    infer_variant_8('test', config_dict['VARIANT_EIGHT_TEST_PROB_PATH'.lower()], need_feature_only=False)
    print('-' * 64)


def infer_message_classifier(config_dict):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)

    model.load_state_dict(torch.load(MESSAGE_MODEL_PATH))
    model.eval()

    if dataset_name == config.SAP_DATASET_NAME:
        messages, labels, urls = message_classifier.read_sap_dataset(need_urls=True)

        message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=0.20, random_state=109)
        
        url_train, url_test, _, _ = train_test_split(urls, [0]*len(urls), test_size=0.20, random_state=109)

    else:
        message_train, message_test, label_train, label_test, url_train, url_test = message_classifier.read_tensor_flow_dataset(dataset_name, need_url_data=True)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Converting using pretrained...")

    train_features = message_classifier.get_roberta_features(tokenizer, message_train)
    test_features = message_classifier.get_roberta_features(tokenizer, message_test)

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

    training_set = message_classifier.TextDataset(partition['train'], labels, id2input, id2mask)
    training_generator = torch.utils.data.DataLoader(training_set, **message_classifier.test_params)

    testing_set = message_classifier.TextDataset(partition['test'], labels, id2input, id2mask)
    testing_generator = torch.utils.data.DataLoader(testing_set, **message_classifier.test_params)


    precision, recall, f1, y_train_probs = message_classifier.predict_test_data(model, training_generator, device, need_probs=True)
    
    with open(config_dict['message_train_prob_path'], 'w') as file:
        writer = csv.writer(file)
        for i, prob in enumerate(y_train_probs):
            writer.writerow([url_train[i], prob])

    precision, recall, f1, y_test_probs = message_classifier.predict_test_data(model, testing_generator, device, need_probs=True)
    
    with open(config_dict['message_test_prob_path'], 'w') as file:
        writer = csv.writer(file)
        for i, prob in enumerate(y_test_probs):
            writer.writerow([url_test[i], prob])


def infer_issue_classifier(config_dict):

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)

    model.load_state_dict(torch.load(ISSUE_MODEL_PATH))
    model.eval()

    if dataset_name == config.SAP_DATASET_NAME:
        texts, labels, urls = issue_classifier.read_sap_issue(need_urls=True)
        text_train, text_test, label_train, label_test = train_test_split(texts, labels, test_size=0.20, random_state=109)
        url_train, url_test, _, _ = train_test_split(urls, [0]*len(urls), test_size=0.20, random_state=109)
    else:
        text_train, text_test, label_train, label_test, url_train, url_test = issue_classifier.read_tensor_flow_issue(dataset_name=dataset_name, need_url_data=True)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Converting using pretrained...")

    train_features = message_classifier.get_roberta_features(tokenizer, text_train, length=256)
    test_features = message_classifier.get_roberta_features(tokenizer, text_test, length=256)

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

    training_set = message_classifier.TextDataset(partition['train'], labels, id2input, id2mask)
    training_generator = torch.utils.data.DataLoader(training_set, **issue_classifier.test_params)

    testing_set = message_classifier.TextDataset(partition['test'], labels, id2input, id2mask)
    testing_generator = torch.utils.data.DataLoader(testing_set, **issue_classifier.test_params)

    precision, recall, f1, y_train_probs = message_classifier.predict_test_data(model, training_generator, device, need_probs=True)
    
    with open(config_dict['issue_train_prob_path'], 'w') as file:
        writer = csv.writer(file)
        for i, prob in enumerate(y_train_probs):
            writer.writerow([url_train[i], prob])

    precision, recall, f1, y_test_probs = message_classifier.predict_test_data(model, testing_generator, device, need_probs=True)
    
    with open(config_dict['issue_test_prob_path'], 'w') as file:
        writer = csv.writer(file)
        for i, prob in enumerate(y_test_probs):
            writer.writerow([url_test[i], prob])


def read_prob_from_file(file_path):
    df = pd.read_csv(file_path, header=None)
    url_to_prob = {}
    for item in df.values.tolist():
        url_to_prob[item[0]] = float(item[1])
    
    return url_to_prob


def commit_classifier_ensemble_new(config_dict):
    print("Reading...")
    url_to_mes_train_prob = read_prob_from_file(config_dict['message_train_prob_path'])
    url_to_mes_test_prob = read_prob_from_file(config_dict['message_test_prob_path'])

    url_to_issue_train_prob = read_prob_from_file(config_dict['issue_train_prob_path'])
    url_to_issue_test_prob = read_prob_from_file(config_dict['issue_test_prob_path'])

    url_to_patch_train_prob = read_prob_from_file(config_dict['patch_train_prob_path'])
    url_to_patch_test_prob = read_prob_from_file(config_dict['patch_test_prob_path'])

    print("Loading dataset info...")
    id_train, id_to_train_label, id_to_train_url = get_dataset_info('train')
    id_test, id_to_test_label, id_to_test_url = get_dataset_info('test')

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for id in id_train:
        url = id_to_train_url[id]
        if url in url_to_mes_train_prob:
            X_train.append([url_to_mes_train_prob[url], url_to_issue_train_prob[url], url_to_patch_train_prob[url]])
            Y_train.append(id_to_train_label[id])
    
    for id in id_test:
        url = id_to_test_url[id]
        if url in url_to_mes_test_prob:
            X_test.append([url_to_mes_test_prob[url], url_to_issue_test_prob[url], url_to_patch_test_prob[url]])
            Y_test.append(id_to_test_label[id])

    print("Training")
    ensemble_classifier = LogisticRegression()

    ensemble_classifier.fit(X=X_train, y=Y_train)

    y_pred = ensemble_classifier.predict(X=X_test)

    f1 = metrics.f1_score(y_pred=y_pred, y_true=Y_test)

    print("F1 Commit ensemble Classifier: {}".format(f1))

    model_path = config_dict['commit_classifier_model_path']

    pickle.dump(ensemble_classifier, open(model_path, 'wb'))


def commit_classifier_ensemble(config_dict):
    url_to_mes_train_prob = read_prob_from_file(config_dict['message_train_prob_path'])
    url_to_mes_test_prob = read_prob_from_file(config_dict['message_test_prob_path'])

    url_to_issue_train_prob = read_prob_from_file(config_dict['issue_train_prob_path'])
    url_to_issue_test_prob = read_prob_from_file(config_dict['issue_test_prob_path'])

    url_to_patch_train_prob = read_prob_from_file(config_dict['ensemble_train_prob_path'])
    url_to_patch_test_prob = read_prob_from_file(config_dict['ensemble_test_prob_path'])

    id_train, id_to_train_label, id_to_train_url = get_dataset_info('train')
    id_test, id_to_test_label, id_to_test_url = get_dataset_info('test')

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for id in id_train:
        url = id_to_train_url[id]
        X_train.append([url_to_mes_train_prob[url], url_to_issue_train_prob[url], url_to_patch_train_prob[url]])
        Y_train.append(id_to_train_label[id])
    
    for id in id_test:
        url = id_to_test_url[id]
        X_test.append([url_to_mes_test_prob[url], url_to_issue_test_prob[url], url_to_patch_test_prob[url]])
        Y_test.append(id_to_test_label[id])

    ensemble_classifier = LogisticRegression()

    ensemble_classifier.fit(X=X_train, y=Y_train)

    y_pred = ensemble_classifier.predict(X=X_test)

    f1 = metrics.f1_score(y_pred=y_pred, y_true=Y_test)

    print("F1 Commit ensemble Classifier: {}".format(f1))

    model_path = config_dict['commit_classifier_model_path']

    pickle.dump(ensemble_classifier, open(model_path, 'wb'))
    

def is_tp(prob, label):
    return prob >= 0.5 and label == 1


def visualize_result(config_dict):

    url_to_mes_test_prob = read_prob_from_file(config_dict['message_test_prob_path'])

    url_to_issue_test_prob = read_prob_from_file(config_dict['issue_test_prob_path'])

    url_to_patch_test_prob = read_prob_from_file(config_dict['patch_test_prob_path'])

    id_test, id_to_test_label, id_to_test_url = get_dataset_info('test')


    message_tp = set()
    issue_tp = set()
    patch_tp = set()

    count_pos = 0
    for id, url in id_to_test_url.items():
        label = id_to_test_label[id]
        if label == 1:
            count_pos += 1

        mes_prob = url_to_mes_test_prob[url]
        if is_tp(mes_prob, label):
            message_tp.add(url)

        issue_prob = url_to_issue_test_prob[url]
        if is_tp(issue_prob, label):
            issue_tp.add(url)

        patch_prob = url_to_patch_test_prob[url]
        if is_tp(patch_prob, label):
            patch_tp.add(url)


    venn_out = venn3([message_tp, issue_tp, patch_tp], ('Message Classifier', 'Issue Classifier', 'Patch Classifier'))
    
    for text in venn_out.set_labels:
        text.set_fontsize(16)
    for text in venn_out.subset_labels:
        text.set_fontsize(16)

    plt.title("Venn diagram for true positive cases", fontsize=16)
    plt.savefig(config_dict['result_visualization_path'])

    print("Total vuln fixes: {}".format(count_pos))

    total_tp = set()
    total_tp.update(message_tp)
    total_tp.update(issue_tp)
    total_tp.update(patch_tp)

    print("Predicted vuln fixes: {}".format(len(total_tp)))


def set_up_config(config_dict):
    global VARIANT_ONE_EMBEDDINGS_DIRECTORY, VARIANT_TWO_EMBEDDINGS_DIRECTORY, VARIANT_THREE_EMBEDDINGS_DIRECTORY
    global VARIANT_FIVE_EMBEDDINGS_DIRECTORY, VARIANT_SIX_EMBEDDINGS_DIRECTORY, VARIANT_SEVEN_EMBEDDINGS_DIRECTORY, VARIANT_EIGHT_EMBEDDINGS_DIRECTORY
    global MESSAGE_MODEL_PATH, ISSUE_MODEL_PATH
    global VARIANT_ONE_MODEL_PATH, VARIANT_TWO_MODEL_PATH, VARIANT_THREE_MODEL_PATH
    global VARIANT_FIVE_MODEL_PATH, VARIANT_SIX_MODEL_PATH, VARIANT_SEVEN_MODEL_PATH, VARIANT_EIGHT_MODEL_PATH

    VARIANT_ONE_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_ONE_EMBEDDINGS_DIRECTORY'.lower()]
    VARIANT_TWO_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_TWO_EMBEDDINGS_DIRECTORY'.lower()]
    VARIANT_THREE_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_THREE_EMBEDDINGS_DIRECTORY'.lower()]
    VARIANT_FIVE_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_FIVE_EMBEDDINGS_DIRECTORY'.lower()]
    VARIANT_SIX_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_SIX_EMBEDDINGS_DIRECTORY'.lower()]
    VARIANT_SEVEN_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_SEVEN_EMBEDDINGS_DIRECTORY'.lower()]
    VARIANT_EIGHT_EMBEDDINGS_DIRECTORY = config_dict['VARIANT_EIGHT_EMBEDDINGS_DIRECTORY'.lower()]
    MESSAGE_MODEL_PATH = config_dict['MESSAGE_MODEL_PATH'.lower()]
    ISSUE_MODEL_PATH = config_dict['ISSUE_MODEL_PATH'.lower()]
    VARIANT_ONE_MODEL_PATH = config_dict['VARIANT_ONE_MODEL_PATH'.lower()]
    VARIANT_TWO_MODEL_PATH = config_dict['VARIANT_TWO_MODEL_PATH'.lower()]
    VARIANT_THREE_MODEL_PATH = config_dict['VARIANT_THREE_MODEL_PATH'.lower()]
    VARIANT_FIVE_MODEL_PATH = config_dict['VARIANT_FIVE_MODEL_PATH'.lower()]
    VARIANT_SIX_MODEL_PATH = config_dict['VARIANT_SIX_MODEL_PATH'.lower()]
    VARIANT_SEVEN_MODEL_PATH = config_dict['VARIANT_SEVEN_MODEL_PATH'.lower()]
    VARIANT_EIGHT_MODEL_PATH = config_dict['VARIANT_EIGHT_MODEL_PATH'.lower()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file',
                        type=str,
                        required=True,
                        help='name of config file')
    args = parser.parse_args()
    config_file_name = args.config_file
    config_parser = configparser.RawConfigParser()
    config_parser.read(config_file_name) 
    config_dict = dict(config_parser.items('DATASET_CONFIG'))

    dataset_name = config_dict['dataset_name']

    set_up_config(config_dict)
    # infer_all_variant(config_dict)
    # get_combined_ensemble_model()

    infer_message_classifier(config_dict)
    infer_issue_classifier(config_dict)

    # commit_classifier_ensemble(config_dict)

    # new function after replacing patch classifier
    commit_classifier_ensemble_new(config_dict)

    # visualize_result(config_dict)

   