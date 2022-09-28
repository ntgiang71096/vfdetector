import json
import os
import pandas as pd
from tqdm import tqdm
import config
# from variant_5_finetune import get_tensor_flow_data
import variant_8_finetune_separate

def line_empty(line):
    if line.strip() == '':
        return True
    else:
        return False

        
def get_line_from_code(sep_token, code):
    lines = []
    for line in code.split('\n'):
        if not line_empty(line):
            lines.append(sep_token + line)

    return lines

def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '/*', '*', '*/', '#')):
                continue
            code = code + line + '\n'

    return code


def get_data_from_saved_file(file_info_name, need_pl=False):
    with open(file_info_name, 'r') as reader:
        data = json.loads(reader.read())

    if need_pl:
        return data['url_data'], data['label_data'], data['url_to_pl'], data['url_to_label']
    else:
        return data['url_data'], data['label_data']


def get_sap_data(dataset_name):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'PL', 'label']]
    items = df.to_numpy().tolist()

    url_train, url_test = [], []
    label_train, label_test = [], []
    url_to_pl = {}
    url_to_label = {}
    for item in tqdm(items):
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        pl = item[3]
        label = item[4]
        url_to_pl[url] = pl
        url_to_label[url] = label
        if partition == 'train':
            if url not in url_train:
                url_train.append(url)
                label_train.append(label)
        elif partition == 'test':
            if url not in url_test:
                url_test.append(url)
                label_test.append(label)
        else:
            Exception("Invalid partition: {}".format(partition))

    print("Finish reading dataset")

    return url_train, url_test, label_train, label_test, url_to_label, url_to_pl


def get_tensor_flow_data(dataset_name):
    # print("Reading dataset...")
    # df = pd.read_csv(dataset_name)
    # df[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition']]
    # items = df.to_numpy().tolist()

    # url_train, url_test = [], []
    # label_train, label_test = [], []
    # url_to_pl = {}
    # url_to_label = {}
    # for item in tqdm(items):
    #     commit_id = item[0]
    #     repo = item[1]
    #     url = repo + '/commit/' + commit_id
    #     partition = item[6]
    #     pl = 'UNKNOWN'
    #     label = item[5]
    #     diff = item[4]

    #     if pd.isnull(diff):   
    #         continue

    #     url_to_pl[url] = pl
    #     url_to_label[url] = label
    #     if partition == 'train':
    #         if url not in url_train:
    #             url_train.append(url)
    #             label_train.append(label)
    #     elif partition == 'test':
    #         if url not in url_test:
    #             url_test.append(url)
    #             label_test.append(label)
    #     else:
    #         Exception("Invalid partition: {}".format(partition))

    # print("Finish reading dataset")

    patch_data, label_data, url_data = variant_8_finetune_separate.get_data(dataset_name)

    url_train, url_test, label_train, label_test, url_to_label, url_to_pl = url_data['train'], url_data['test'], label_data['train'], label_data['test'], {}, {}

    return url_train, url_test, label_train, label_test, url_to_label, url_to_pl


def get_data(dataset_name, need_pl=False):
    file_info_name = 'info_' + dataset_name + '.json'
    if os.path.isfile(file_info_name):
        return get_data_from_saved_file(file_info_name, need_pl)

    if dataset_name == config.SAP_DATASET_NAME:
        print("load sap dataset...")
        # Need to read csv version instead
        dataset_name = 'sap_patch_dataset.csv'
        url_train, url_test, label_train, label_test, url_to_label, url_to_pl = get_sap_data(dataset_name)

    else:
        url_train, url_test, label_train, label_test, url_to_label, url_to_pl = get_tensor_flow_data(dataset_name)

    url_data = {'train': url_train, 'test': url_test}
    label_data = {'train': label_train, 'test': label_test}

    data = {'url_data': url_data, 'label_data': label_data, 'url_to_pl': url_to_pl, 'url_to_label' : url_to_label}

    json.dump(data, open(file_info_name, 'w'))

    if need_pl:
        return url_data, label_data, url_to_pl, url_to_label
    else:
        return url_data, label_data