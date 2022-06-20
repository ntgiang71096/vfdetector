import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import issue_classifier 
import variant_ensemble


plt.rc('axes', unicode_minus=False)


def get_max_values(id_list, id_to_features):
    max_values = []
    for id in id_list:
        values = [item[1] for item in id_to_features[str(id)]]
        max_values.append(abs(max(values, key=abs)))

    return max_values


def get_feature_count(id_to_features, id_list):
    threshold = 0.1
    pos_word_to_count = {}
    neg_word_to_count = {}
    for id in id_list:
        for word, value in id_to_features[str(id)]:
            # ignore outlier
            if word.lower() in ['when', 'the', 'an', 'in']:
                continue

            word = word.lower()
            if value >= threshold:
                if word.lower() not in pos_word_to_count:
                    pos_word_to_count[word] = 0
                pos_word_to_count[word] += 1
            elif abs(value) > threshold:
                if word.lower() not in neg_word_to_count:
                    neg_word_to_count[word] = 0
                neg_word_to_count[word] += 1
    pos_list = [(word, count) for word, count in pos_word_to_count.items()]
    pos_list.sort(key=lambda x: x[1], reverse=True)

    neg_list = [(word, count) for word, count in neg_word_to_count.items()]
    neg_list.sort(key=lambda x: x[1], reverse=True)

    return pos_list, neg_list


def get_occ(vuln_terms, neg_test):
    term_to_count = {}
    list_msg_tokens = []
    for msg in neg_test:
        list_msg_tokens.append(msg.split(' '))

    for term in vuln_terms:
        count = 0
        for tokens in list_msg_tokens:
            if term in tokens:
                count += 1
        term_to_count[term] = count

    term_occ = [(term, count) for term, count in term_to_count.items()]
    term_occ.sort(key=lambda x: x[1], reverse=True)

    return term_occ


def visualize():

    texts, labels = issue_classifier.read_sap_issue()

    text_train, text_test, label_train, label_test = train_test_split(texts, labels, test_size=0.20, random_state=109)
    
    pos_ids, neg_ids = [], []
    pos_text, neg_text = [], []
    for i, label in enumerate(label_test):
        if text_test[i] in ['', '...', ' ']:          # ignore blank messages
            continue
        if label == 1:
            pos_ids.append(i)
        else:
            neg_ids.append(i)

    url_to_issue_test_prob = variant_ensemble.read_prob_from_file('probs/issue_prob_test.txt')
    id_test, id_to_test_label, id_to_test_url = variant_ensemble.get_dataset_info('test')
    
    tp, tn, fp, fn = [], [], [], []
    
    neg_test = []

    for id, url in id_to_test_url.items():
        if id in neg_ids:
            neg_test.append(text_test[id])

        y_pred = 1 if url_to_issue_test_prob[url] >= 0.5 else 0
        if id in pos_ids and y_pred == 1:
            tp.append(id)
        elif id in pos_ids and y_pred == 0:
            fp.append(id)
        elif id in neg_ids and y_pred == 0:
            tn.append(id)
        elif id in neg_ids and y_pred == 1:
            fn.append(id)

  
    # df = pd.read_csv('ffmpeg_predictions.csv')
    # for item in df.values.tolist():
    #     id = item[0]
    #     y_pred = item[1]
    #     y_test = item[2]

    #     if y_test == 0:
    #         neg_test.append(test_msg[id])

    #     if y_pred == y_test == 1:
    #         tp.append(id)
    #     elif y_pred == y_test == 0:
    #         tn.append(id)
    #     elif y_pred == 1:
    #         fp.append(id)
    #     else:
    #         fn.append(id)

    # print("Len Neg Test: {}".format(len(neg_test)))


    with open('issue_explanation.json', 'r') as file:
        id_to_features = json.load(file)

    tp_max_values = get_max_values(tp, id_to_features)
    tn_max_values = get_max_values(tn, id_to_features)
    fp_max_values = get_max_values(fp, id_to_features)
    fn_max_values = get_max_values(fn, id_to_features)

    data = [tp_max_values, tn_max_values, fp_max_values, fn_max_values]

    tp_pos_count, tp_neg_count = get_feature_count(id_to_features, tp)
    tn_pos_count, tn_neg_count = get_feature_count(id_to_features, tn)

    # tp_pos_count, tp_neg_count, tn_pos_count, tn_neg_count = tp_pos_count[:15], tp_neg_count[:15], tn_pos_count[:15], tn_neg_count[:15]

    vuln_terms = [item[0] for item in tp_pos_count]
    vuln_terms = vuln_terms[:20]

    vuln_occ = get_occ(vuln_terms, neg_test)


    plt.figure()
    plt.rc('axes', unicode_minus=False)
    plt.rcParams["figure.figsize"] = (9, 4)
    plt.barh(*zip(*vuln_occ))
    plt.title("Fig 6. Vulnerability-related terms' occurrence in Non-vulnerability-fixing commits")
    plt.xlabel("Occurrence")
    plt.savefig('imgs/issue_vuln_in_non_vuln_commit.png')
    plt.close()


    tp_pos_count = tp_pos_count[:20]
    plt.figure()
    plt.rc('axes', unicode_minus=False)
    plt.barh(*zip(*tp_pos_count))
    plt.title("Fig 2. Positive terms' occurrence for true positive cases")
    plt.xlabel("Occurrence")
    plt.savefig('imgs/issue_tp_pos_count.png')
    plt.close()


    tp_neg_count = tp_neg_count[:20]
    plt.figure()
    plt.rc('axes', unicode_minus=False)
    plt.barh(*zip(*tp_neg_count))
    plt.title("Fig 3. Negative terms' occurrence for true positive cases")
    plt.xlabel("Occurrence")
    plt.savefig('imgs/issue_tp_neg_count.png')
    plt.close()


    tn_pos_count = tn_pos_count[:20]
    plt.figure()
    plt.rc('axes', unicode_minus=False)
    plt.barh(*zip(*tn_pos_count))
    plt.title("Fig 4. Positive terms' occurrence for true negative cases")
    plt.xlabel("Occurrence")
    plt.savefig('imgs/issue_tn_pos_count.png')
    plt.close()


    tn_neg_count = tn_neg_count[:20]
    plt.figure()
    plt.rc('axes', unicode_minus=False)
    plt.barh(*zip(*tn_neg_count))
    plt.title("Fig 5. Negative terms' occurrence for true negative cases")
    plt.xlabel("Occurrence")
    plt.savefig('imgs/issue_tn_neg_count.png')
    plt.close()


    plt.figure()
    plt.rc('axes', unicode_minus=False)
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.boxplot(data, labels=['True Positive', 'True Negative', 'False Positive', 'False Negative'])
    plt.title('Fig 2. Highest weight\'s term distribution on issue classifier')
    plt.ylabel('Highest weight')
    plt.xlabel('Segment')
    plt.savefig('imgs/issue_weight_plot.png')
    plt.close()


if __name__ == '__main__':
    visualize()