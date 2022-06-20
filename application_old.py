from copyreg import pickle
from pydoc import classname
from model import *
import importlib
from torch import cuda
import utils
import pandas as pd
import variant_6_finetune
import preprocess_finetuned_variant_1
import preprocess_finetuned_variant_2
import preprocess_finetuned_variant_3
import preprocess_finetuned_variant_5
import preprocess_finetuned_variant_6
import preprocess_finetuned_variant_7
import preprocess_finetuned_variant_8
import patch_entities
from model import EnsembleModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import message_classifier
import pickle
import json

dataset_name = 'tf_vuln_dataset.csv'

variant_one_finetuned_model_path = 'model/tf_patch_variant_1_finetuned_model.sav'
variant_two_finetuned_model_path = 'model/tf_patch_variant_2_finetuned_model.sav'
variant_three_finetuned_model_path = 'model/tf_patch_variant_3_finetuned_model.sav'
variant_five_finetuned_model_path = 'model/tf_patch_variant_5_finetuned_model.sav'
variant_six_finetuned_model_path = 'model/tf_patch_variant_6_finetuned_model.sav'
variant_seven_finetuned_model_path = 'model/tf_patch_variant_7_finetuned_model.sav'
variant_eight_finetuned_model_path = 'model/tf_patch_variant_8_finetuned_model.sav'

variant_one_model_path = 'model/tf_patch_variant_1_model.sav'
variant_two_model_path = 'model/tf_patch_variant_2_model.sav'
variant_three_model_path = 'model/tf_patch_variant_3_model.sav'
variant_five_model_path = 'model/tf_patch_variant_5_model.sav'
variant_six_model_path = 'model/tf_patch_variant_6_model.sav'
variant_seven_model_path = 'model/tf_patch_variant_7_model.sav'
variant_eight_model_path = 'model/tf_patch_variant_8_model.sav'
patch_ensemble_model_path = 'model/tf_patch_ensemble.sav'

message_model_path = 'model/tf_message_classifier.sav'
issue_model_path = 'model/tf_issue_classifier.sav'

commit_classifier_model_path = 'model/tf_commit_classifier.sav'

codebert_1, codebert_2, codebert_3, codebert_5, codebert_6, codebert_7, codebert_8 = None, None, None, None, None, None, None
model_1, model_2, model_3, model_5, model_6, model_7, model_8 = None, None, None, None, None, None, None
patch_ensemble_model = None

tokenizer = None

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def load_codebert(class_name, model_path):
    m = importlib.__import__('model')
    model_class = getattr(m, class_name)
    model = model_class()
    # model = VariantOneFinetuneClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path))
    code_bert = model.module.code_bert
    code_bert.eval()

    return code_bert

def load_model(class_name, model_path):
    m = importlib.__import__('model')
    model_class = getattr(m, class_name)
    model = model_class()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.module.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def get_variant_1_features(code_changes):
    # print('Loading code_bert for variant one...')
    codebert_1 = load_codebert('VariantOneFinetuneClassifier', variant_one_finetuned_model_path)
    model_1 = load_model('VariantOneClassifier', variant_one_model_path)
    # print('Finish loading')

    code = ''
    for item in code_changes:
        code = code + item + '\n'

    removed_code = preprocess_finetuned_variant_1.get_code_version(code, False)
    added_code = preprocess_finetuned_variant_1.get_code_version(code, True)

    code = removed_code + tokenizer.sep_token + added_code

    codebert_1.to(device)
    model_1.to(device)

    embeddings = preprocess_finetuned_variant_1.get_commit_embeddings([code], tokenizer, codebert_1)
    embeddings = torch.FloatTensor(embeddings)
    features = model_1(embedding_batch=embeddings, need_final_feature=True)[1][0]
    
    # codebert_1.to('cpu')
    # model_1.to('cpu')

    del model_1
    del codebert_1
    
    return features


def get_variant_2_features(code_changes):

    # print('Loading code_bert for variant two...')
    codebert_2 = load_codebert('VariantTwoFineTuneClassifier', variant_two_finetuned_model_path)
    model_2 = load_model('VariantTwoClassifier', variant_two_model_path)
    # print('Finish loading')

    code_list = []
    for diff in code_changes:
        removed_code = preprocess_finetuned_variant_2.get_code_version(diff, False)
        added_code = preprocess_finetuned_variant_2.get_code_version(diff, True)
        code = removed_code + tokenizer.sep_token + added_code
        code_list.append(code)
    
    codebert_2.to(device)
    model_2.to(device)

    file_embeddings = preprocess_finetuned_variant_2.get_file_embeddings(code_list, tokenizer, codebert_2)
    if len(file_embeddings) > 5:
            file_embeddings = file_embeddings[:5]
    while len(file_embeddings) < 5:
        file_embeddings.append(patch_entities.empty_embedding)

    file_embeddings = torch.FloatTensor(file_embeddings)
    file_embeddings = torch.unsqueeze(file_embeddings, 0)
    features = model_2(file_batch=file_embeddings, need_final_feature=True)[1][0]

    codebert_2.to('cpu')
    model_2.to('cpu')
    
    del model_2
    del codebert_2
    
    return features


def get_variant_3_features(code_changes):

    # print('Loading code_bert for variant three...')
    codebert_3 = load_codebert('VariantThreeFineTuneOnlyClassifier', variant_three_finetuned_model_path)
    model_3 = load_model('VariantThreeClassifier', variant_three_model_path)
    # print('Finish loading')

    embeddings = []
    hunk_list = []

    for item in code_changes:
        hunk_list.extend(preprocess_finetuned_variant_3.get_hunk_from_diff(item))

    code_list =[]

    for hunk in hunk_list:
        removed_code = preprocess_finetuned_variant_3.get_code_version(hunk, False)
        added_code = preprocess_finetuned_variant_3.get_code_version(hunk, True)

        code = removed_code + tokenizer.sep_token + added_code
        code_list.append(code)
    
    codebert_3.to(device)
    model_3.to(device)

    embeddings = preprocess_finetuned_variant_3.get_hunk_embeddings(code_list, tokenizer, codebert_3)
    while len(embeddings) < 5:
        embeddings.append([0] * len(embeddings[0]))

    embeddings = torch.FloatTensor(embeddings)
    embeddings = torch.unsqueeze(embeddings, 0)
    features = model_3(code=embeddings, need_final_feature=True)[1][0]

    # codebert_3.to('cpu')
    # model_3.to('cpu')

    del codebert_3
    del model_3

    return features

def get_variant_5_features(code_changes):

    # print('Loading code_bert for variant five...')
    codebert_5 = load_codebert('VariantFiveFineTuneClassifier', variant_five_finetuned_model_path)
    model_5 = load_model('VariantFiveClassifier', variant_five_model_path)
    # print('Finish loading')

    code = ''
    for item in code_changes:
        code = code + item + '\n'

    removed_code = tokenizer.sep_token + preprocess_finetuned_variant_5.get_code_version(code, False)
    added_code = tokenizer.sep_token + preprocess_finetuned_variant_5.get_code_version(code, True)

    codebert_5.to(device)
    model_5.to(device)

    removed_embeddings = preprocess_finetuned_variant_5.get_commit_embeddings([removed_code], tokenizer, codebert_5)
    added_embeddings = preprocess_finetuned_variant_5.get_commit_embeddings([added_code], tokenizer, codebert_5)
    removed_embeddings = torch.FloatTensor(removed_embeddings)
    added_embeddings = torch.FloatTensor(added_embeddings)

    features = model_5(before_batch=removed_embeddings, after_batch=added_embeddings, need_final_feature=True)[1][0]

    # codebert_5.to('cpu')
    # model_5.to('cpu')

    del codebert_5
    del model_5

    return features


def get_variant_6_features(code_changes):

    # print('Loading code_bert for variant six...')
    codebert_6 = load_codebert('VariantSixFineTuneClassifier', variant_six_finetuned_model_path)
    model_6 = load_model('VariantSixClassifier', variant_six_model_path)
    # print('Finish loading')

    removed_code_list = []
    added_code_list = []
    
    for diff in code_changes:
        removed_code = tokenizer.sep_token + preprocess_finetuned_variant_6.get_code_version(diff, False)
        added_code = tokenizer.sep_token + preprocess_finetuned_variant_6.get_code_version(diff, True)

        removed_code_list.append(removed_code)
        added_code_list.append(added_code)

    codebert_6.to(device)
    model_6.to(device)

    removed_embeddings = preprocess_finetuned_variant_6.get_file_embeddings(removed_code_list, tokenizer, codebert_6)
    added_embeddings = preprocess_finetuned_variant_6.get_file_embeddings(added_code_list, tokenizer, codebert_6)

    if len(removed_embeddings) > 5:
            removed_embeddings = removed_embeddings[:5]
    if len(added_embeddings) > 5:
        added_embeddings = added_embeddings[:5]
    while len(removed_embeddings) < 5:
        removed_embeddings.append(patch_entities.empty_embedding)
    while len(added_embeddings) < 5:
        added_embeddings.append(patch_entities.empty_embedding)

    removed_embeddings = torch.FloatTensor(removed_embeddings)
    added_embeddings = torch.FloatTensor(added_embeddings)

    removed_embeddings = torch.unsqueeze(removed_embeddings, 0)
    added_embeddings = torch.unsqueeze(added_embeddings, 0)

    features = model_6(before_batch=removed_embeddings, after_batch=added_embeddings, need_final_feature=True)[1][0]

    # codebert_6.to('cpu')
    # model_6.to('cpu')

    del codebert_6
    del model_6

    return features


def get_variant_7_features(code_changes):

    # print('Loading code_bert for variant seven...')
    codebert_7 = load_codebert('VariantSeventFineTuneOnlyClassifier', variant_seven_finetuned_model_path)
    model_7 = load_model('VariantSevenClassifier', variant_seven_model_path)
    # print('Finish loading')

    hunk_list = []

    for item in code_changes:
        hunk_list.extend(preprocess_finetuned_variant_7.get_hunk_from_diff(item))

    removed_code_list =[]
    added_code_list = []

    has_removed_code = False
    has_added_code = False

    for hunk in hunk_list:
        removed_code = preprocess_finetuned_variant_7.get_code_version(hunk, False)
        if removed_code.strip() != '':
            removed_code_list.append(removed_code)
            has_removed_code = True

        added_code = preprocess_finetuned_variant_7.get_code_version(hunk, True)
        if added_code.strip() != '':
            added_code_list.append(added_code)
            has_added_code = True
    
    if not has_removed_code:
        removed_code_list.append('')
    
    if not has_added_code:
        added_code_list.append('')

    codebert_7.to(device)
    model_7.to(device)

    removed_embeddings = preprocess_finetuned_variant_7.get_hunk_embeddings(removed_code_list, tokenizer, codebert_7)
    added_embeddings = preprocess_finetuned_variant_7.get_hunk_embeddings(added_code_list, tokenizer, codebert_7)
    
    while len(removed_embeddings) < 5:
        removed_embeddings.append([0] * len(removed_embeddings[0]))

    while len(added_embeddings) < 5:
            added_embeddings.append([0] * len(added_embeddings[0]))

    removed_embeddings = torch.FloatTensor(removed_embeddings)
    removed_embeddings = torch.unsqueeze(removed_embeddings, 0)

    added_embeddings = torch.FloatTensor(added_embeddings)
    added_embeddings = torch.unsqueeze(added_embeddings, 0)

    features = model_7(before_batch=removed_embeddings, after_batch=added_embeddings, need_final_feature=True)[1][0]

    codebert_7.to('cpu')
    model_7.to('cpu')

    del codebert_7
    del model_7

    return features


def get_variant_8_features(code_changes):

    # print('Loading code_bert for variant eight...')
    codebert_8 = load_codebert('VariantEightFineTuneOnlyClassifier', variant_eight_finetuned_model_path)
    model_8 = load_model('VariantEightClassifier', variant_eight_model_path)
    # print('Finish loading')

    removed_code_list = []
    added_code_list = []

    for diff in code_changes:
        removed_code = preprocess_finetuned_variant_8.get_code_version(diff, False)
        added_code = preprocess_finetuned_variant_8.get_code_version(diff, True)

        new_removed_code_list = preprocess_finetuned_variant_8.get_line_from_code(tokenizer.sep_token, removed_code)
        new_added_code_list = preprocess_finetuned_variant_8.get_line_from_code(tokenizer.sep_token, added_code)

        if len(new_removed_code_list) == 0:
            new_removed_code_list = [tokenizer.sep_token]
        
        if len(new_added_code_list) == 0:
            new_added_code_list = [tokenizer.sep_token]
        
        removed_code_list.extend(new_removed_code_list)
        added_code_list.extend(new_added_code_list)
    
    codebert_8.to(device)
    model_8.to(device)

    removed_embeddings = preprocess_finetuned_variant_8.get_line_embeddings(removed_code_list, tokenizer, codebert_8)
    added_embeddings = preprocess_finetuned_variant_8.get_line_embeddings(added_code_list, tokenizer, codebert_8)

    while len(removed_embeddings) < 5:
        removed_embeddings.append([0] * len(removed_embeddings[0]))

    while len(added_embeddings) < 5:
            added_embeddings.append([0] * len(added_embeddings[0]))

    removed_embeddings = torch.FloatTensor(removed_embeddings)
    removed_embeddings = torch.unsqueeze(removed_embeddings, 0)

    added_embeddings = torch.FloatTensor(added_embeddings)
    added_embeddings = torch.unsqueeze(added_embeddings, 0)

    features = model_8(before_batch=removed_embeddings, after_batch=added_embeddings, need_final_feature=True)[1][0]

    # codebert_8.to('cpu')
    # model_8.to('cpu')

    # print(features.shape)

    del codebert_8
    del model_8

    return features

def retrieve_features(code_changes_list):
    features = []

    print("Extracting variant 1 features")
    variant_1_features = []
    for code in code_changes_list:
        variant_1_features.append(get_variant_1_features(code))
    variant_1_features = torch.stack(variant_1_features)

    features.append(variant_1_features)

    print("Extracting variant 2 features")
    variant_2_features = []
    for code in code_changes_list:
        variant_2_features.append(get_variant_2_features(code))
    variant_2_features = torch.stack(variant_2_features)

    features.append(variant_2_features)

    print("Extracting variant 3 features")
    variant_3_features = []
    for code in code_changes_list:
        variant_3_features.append(get_variant_3_features(code))
    variant_3_features = torch.stack(variant_3_features)

    features.append(variant_3_features)

    print("Extracting variant 5 features")
    variant_5_features = []
    for code in code_changes_list:
        variant_5_features.append(get_variant_5_features(code))
    variant_5_features = torch.stack(variant_5_features)

    features.append(variant_5_features)

    print("Extracting variant 6 features")
    variant_6_features = []
    for code in code_changes_list:
        variant_6_features.append(get_variant_6_features(code))
    variant_6_features = torch.stack(variant_6_features)

    features.append(variant_6_features)

    print("Extracting variant 7 features")
    variant_7_features = []
    for code in code_changes_list:
        variant_7_features.append(get_variant_7_features(code))
    variant_7_features = torch.stack(variant_7_features)

    features.append(variant_7_features)

    print("Extracting variant 8 features")
    variant_8_features = []
    for code in code_changes_list:
        variant_8_features.append(get_variant_8_features(code))
    variant_8_features = torch.stack(variant_8_features)

    features.append(variant_8_features)
    
    return features

def predict_patch(code_changes_list):
    # global codebert_1, codebert_2, codebert_3, codebert_5, codebert_6, codebert_7, codebert_8
    # global model_1, model_2, model_3, model_5, model_6, model_7, model_8
    global tokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # print('Loading code_bert for variant one...')
    # codebert_1 = load_codebert('VariantOneFinetuneClassifier', variant_one_finetuned_model_path)
    # model_1 = load_model('VariantOneClassifier', variant_one_model_path)
    # print('Finish loading')

    # print('Loading code_bert for variant two...')
    # codebert_2 = load_codebert('VariantTwoFineTuneClassifier', variant_two_finetuned_model_path)
    # model_2 = load_model('VariantTwoClassifier', variant_two_model_path)
    # print('Finish loading')

    # print('Loading code_bert for variant three...')
    # codebert_3 = load_codebert('VariantThreeFineTuneOnlyClassifier', variant_three_finetuned_model_path)
    # model_3 = load_model('VariantThreeClassifier', variant_three_model_path)
    # print('Finish loading')

    # print('Loading code_bert for variant five...')
    # codebert_5 = load_codebert('VariantFiveFineTuneClassifier', variant_five_finetuned_model_path)
    # model_5 = load_model('VariantFiveClassifier', variant_five_model_path)
    # print('Finish loading')

    # print('Loading code_bert for variant six...')
    # codebert_6 = load_codebert('VariantSixFineTuneClassifier', variant_six_finetuned_model_path)
    # model_6 = load_model('VariantSixClassifier', variant_six_model_path)
    # print('Finish loading')

    # print('Loading code_bert for variant seven...')
    # codebert_7 = load_codebert('VariantSeventFineTuneOnlyClassifier', variant_seven_finetuned_model_path)
    # model_7 = load_model('VariantSevenClassifier', variant_seven_model_path)
    # print('Finish loading')

    print('Loading code_bert for variant eight...')
    codebert_8 = load_codebert('VariantEightFineTuneOnlyClassifier', variant_eight_finetuned_model_path)
    model_8 = load_model('VariantEightClassifier', variant_eight_model_path)
    print('Finish loading')

    print('Loading patch ensemble classifier...')
    patch_ensemble_model = EnsembleModel()
    if torch.cuda.device_count() > 1:
        patch_ensemble_model = nn.DataParallel(patch_ensemble_model)
    patch_ensemble_model.load_state_dict(torch.load(patch_ensemble_model_path))
    patch_ensemble_model.eval()
  
    print('Finish loading')

    print("Calculating features...")
    feature_list = retrieve_features(code_changes_list)
    print("Done!")
    
    patch_ensemble_model.to(device)
    outs = patch_ensemble_model(feature_list[0], feature_list[1], feature_list[2], feature_list[3], feature_list[4], feature_list[5], feature_list[6])
    outs = F.softmax(outs, dim=1)

    patch_ensemble_model.to('cpu')

    probs = []
    for item in outs.tolist():
        probs.append(item[1])
    
    print(probs)
    # return outs.tolist()[0][1]

def predict_message(message_list):
    print("Loading message classifier...")

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.load_state_dict(torch.load(message_model_path))
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    print("Finish loading")

    print("Infering...")

    inputs = tokenizer(message_list, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    input_ids = inputs.data['input_ids']
    masks = inputs.data['attention_mask']

    # input_ids = torch.unsqueeze(input_ids, 0)
    # masks = torch.unsqueeze(masks, 0)

    input_ids = input_ids.to(device)    
    masks = masks.to(device)

    outs = model(input_ids, masks)

    del model

    # print(torch.argmax(outs.logits, dim=1).tolist())
    return F.softmax(outs.logits, dim=1).tolist()[0][1]


def predict_issue(text_list):
    print("Loading issue classifier...")

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    model.load_state_dict(torch.load(issue_model_path))
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # input_ids, masks = message_classifier.get_roberta_features(tokenizer, [text], length=256)[0]
    
    inputs = tokenizer(text_list, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
    input_ids = inputs.data['input_ids']
    masks = inputs.data['attention_mask']

    print("Finish loading")

    print("Infering...")

    # input_ids = torch.unsqueeze(input_ids, 0)
    # masks = torch.unsqueeze(masks, 0)

    input_ids = input_ids.to(device)    
    masks = masks.to(device)

    outs = model(input_ids, masks)

    # print(torch.argmax(outs.logits, dim=1).tolist())

    del model

    return F.softmax(outs.logits, dim=1).tolist()[0][1]


def predict_ensemble(model, message_prob, issue_prob, patch_prob):
    return model.predict_proba([[message_prob, issue_prob, patch_prob]])[0][1]


def batch_predict(file_path, output_file_path):
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
        if 'issue' in item:
            issue_list.append(item['issue'])
            issue_id.append(id)
        if 'patch' in item:
            patch_list.append(item['patch'])
            patch_id.append(id)


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

    json.dump(output_list, open(output_file_path, 'w'))


def get_code_changes_sample(index):
    patch_data, label_data, url_data = variant_6_finetune.get_data(dataset_name)
    
    code_changes = patch_data['test'][index]
    label = label_data['test'][index]

    for code in code_changes:
        print(json.dumps(code))
        print()
        print('*' * 32)     
        print() 
    print(label)
    return code_changes


if __name__== '__main__':
    probs = predict_issue(['right after install TensorFlowLiteObjC , by pod install, cause error in Xcode'])
    print()
    # predict_message('Prevent memory leak in decoding PNG images. PiperOrigin-RevId: 409300653 Change-Id: I6182124c545989cef80cefd439b659095920763b')
    # print(predict_ensemble(0.9, 0.8, 0.9))
    predict_patch([get_code_changes_sample(1), get_code_changes_sample(5)])
    # get_code_changes_sample(5)
    # batch_predict('sample_1.json', 'prediction_sample_1.json')
    