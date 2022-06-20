from copyreg import pickle
from pydoc import classname
from unittest.mock import patch
from model import *
import importlib
from torch import cuda
import pandas as pd
import patch_entities
from model import VulFixMinerClassifier, VulFixMinerFineTuneClassifier
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pickle
import json
from vulfixminer import CommitAggregator
import issue_linker_infer
import argparse

import logging

logging.disable(logging.WARNING)

dataset_name = 'tf_vuln_dataset.csv'

patch_finetune_model_path = 'model/tf_patch_vulfixminer_finetuned_model.sav'
patch_model_path = 'model/tf_patch_vulfixminer.sav'
message_model_path = 'model/tf_message_classifier.sav'
issue_model_path = 'model/tf_issue_classifier.sav'
commit_classifier_model_path = 'model/tf_commit_classifier.sav'

patch_ensemble_model = None

tokenizer = None

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

message_classifier = None
issue_classifier = None
file_transformer = None
patch_classifier = None


def load_message_classifier():
    print("Loading message classifier...")

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.load_state_dict(torch.load(message_model_path))
    model.eval()

    print("Finish loading")

    return model


def load_issue_classifier():
    print("Loading issue classifier...")

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.load_state_dict(torch.load(issue_model_path))
    model.eval()

    print("Finish loading")

    return model

def load_patch_classifier():
    print('Loading patch classifier...')
    
    finetune_model = VulFixMinerFineTuneClassifier()

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        finetune_model = nn.DataParallel(finetune_model)

    finetune_model.load_state_dict(torch.load(patch_finetune_model_path))
    code_bert = finetune_model.module.code_bert
    code_bert.eval()
    code_bert.to(device)

    model = VulFixMinerClassifier()

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.module.load_state_dict(torch.load(patch_model_path))
    model.to(device)
    model.eval()

    print('Finish loading')

    return model, code_bert


def load_models():
    
    global message_classifier, issue_classifier, file_transformer, patch_classifier
    
    message_classifier = load_message_classifier()

    issue_classifier = load_issue_classifier()

    patch_classifier, file_transformer = load_patch_classifier()

    
def predict_message(message_list):

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # print("Infering...")

    inputs = tokenizer(message_list, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    input_ids = inputs.data['input_ids']
    masks = inputs.data['attention_mask']

    input_ids = input_ids.to(device)    
    masks = masks.to(device)

    outs = message_classifier(input_ids, masks)

    outs = F.softmax(outs.logits, dim=1)

    outs = outs[:, 1].tolist()

    return outs

    
def predict_patch(patch_list):
    aggregator = CommitAggregator(file_transformer)

    embeddings = []
    for patch in patch_list:
        embeddings.append(aggregator.transform(patch))
    
    embeddings = torch.stack(embeddings)

    outs = patch_classifier(embeddings)
    outs = F.softmax(outs, dim=1)

    outs = outs[:, 1].tolist()

    return outs
    

def predict_issue(text_list):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    inputs = tokenizer(text_list, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
    input_ids = inputs.data['input_ids']
    masks = inputs.data['attention_mask']

    # print("Infering...")

    input_ids = input_ids.to(device)    
    masks = masks.to(device)

    outs = issue_classifier(input_ids, masks)

    outs = F.softmax(outs.logits, dim=1)

    outs = outs[:, 1].tolist()

    return outs


def predict_ensemble(model, message_prob, issue_prob, patch_prob):
    return model.predict_proba([[message_prob, issue_prob, patch_prob]])[0][1]


def batch_predict(args):

    MODE = args.mode
    threshold = args.threshold

    print("Using VFDetector in mode: {}".format(MODE))
    if MODE not in ('prediction', 'ranking'):
        raise Exception('mode need to be either prediction or ranking')

    file_path = args.input
    output_file_path = args.output

    with open(file_path, 'r') as reader:
        item_list = json.loads(reader.read())

    message_list = []
    message_id = []
    issue_list = []
    issue_id = []
    # list of code changes
    patch_list = []
    patch_id = []

    id_list = []
    for item in item_list:
        id = item['id']
        id_list.append(id)
        if 'message' in item:
            message_list.append(item['message'])
            message_id.append(id)
        if 'patch' in item:
            patch_list.append(item['patch'])
            patch_id.append(id)
        if 'issue' in item:
            issue_list.append(item['issue'])
            issue_id.append(id)
        else:
            print("Commit with id {} does not have issue report, linking to the most similar one in corpus...".format(id))
            best_issue = issue_linker_infer.infer_issue(id, item['message'], item['patch'])
            print("Linked commit with issue number: {}".format(best_issue['number']))
            issue_str = ''
            issue_str = issue_str + best_issue['title'] + '\n' + best_issue['body'] + '\n'
            for comment in best_issue['comments']:
                issue_str = issue_str + comment + '\n'
            issue_list.append(issue_str)
            issue_id.append(id)

    msg_probs = predict_message(message_list)
    issue_probs = predict_issue(issue_list)
    patch_probs = predict_patch(patch_list)

    id_to_msg_prob = {}
    for i, prob in enumerate(msg_probs):
        id_to_msg_prob[message_id[i]] = prob

    id_to_issue_prob = {}
    for i, prob in enumerate(issue_probs):
        id_to_issue_prob[issue_id[i]] = prob

    id_to_patch_prob = {}
    for i, prob in enumerate(patch_probs):
        id_to_patch_prob[patch_id[i]] = prob

    id_to_commit_prob = {}

    model = pickle.load(open(commit_classifier_model_path, 'rb'))

    for id in id_list:
        if id in id_to_msg_prob and id in id_to_issue_prob and id in id_to_patch_prob:
            id_to_commit_prob[id] = predict_ensemble(model, id_to_msg_prob[id], id_to_issue_prob[id], id_to_patch_prob[id])

    output_list = []
    for id in id_list:
        output = {}
        output['id'] = id
        if id in id_to_msg_prob:
            output['message_prob'] = id_to_msg_prob[id]
        if id in id_to_issue_prob:
            output['issue_prob'] = id_to_issue_prob[id]
        if id in id_to_patch_prob:
            output['patch_prob'] = id_to_patch_prob[id]
        if id in id_to_commit_prob:
            output['commit_prob'] = id_to_commit_prob[id]

        output_list.append(output)

    if MODE == 'prediction':
        result = []
        for id in id_list:
            prob = id_to_commit_prob[id]
            output = 'vulnerability-fixing commit'
            if prob < threshold:
                output = 'non-vulnerability-fixing commit'
            result.append({'id': id, 'prediction':  output})
        
        json.dump(result, open(output_file_path, 'w'))

    elif MODE == 'ranking':
        id_prob_list = []
        for id in id_list:
            id_prob_list.append((id, id_to_commit_prob[id]))
        
        id_prob_list = sorted(id_prob_list, key=lambda x: x[1], reverse=True)
        
        result = []
        for id, prob in id_prob_list:
            result.append({'id': id, 'score': prob})
        
        json.dump(result, open(output_file_path, 'w'))

    else:
        raise('mode need to be either prediction or ranking')

    # json.dump(output_list, open(output_file_path, 'w'))

    print("Finish process!")

# def get_code_changes_sample(index):
#     patch_data, label_data, url_data = variant_6_finetune.get_data(dataset_name)
    
#     code_changes = patch_data['test'][index]
#     label = label_data['test'][index]

#     for code in code_changes:
#         print(json.dumps(code))
#         print()
#         print('*' * 32)     
#         print() 
#     print(label)
#     return code_changes


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-mode',
                        type=str,
                        required=True,
                        help='mode prediction or ranking')
    parser.add_argument('-input',
                        type=str,
                        required=True,
                        help='path to json input file')
    parser.add_argument('-threshold',
                        type=float,
                        required=False,
                        help='threshold for prediction, default 0.5',
                        default=0.5)
    parser.add_argument('-output',
                        type=str,
                        required=True,
                        help='path to json output file')
    args = parser.parse_args()

    # probs = predict_issue(['right after install TensorFlowLiteObjC , by pod install, cause error in Xcode'])
    # print(probs)
    # probs = predict_message('Prevent memory leak in decoding PNG images. PiperOrigin-RevId: 409300653 Change-Id: I6182124c545989cef80cefd439b659095920763b')
    # print(probs)
    # predict_patch([get_code_changes_sample(1), get_code_changes_sample(5)])
    # print(predict_ensemble(0.9, 0.8, 0.9))
    # get_code_changes_sample(5)
    load_models()
    batch_predict(args)
    