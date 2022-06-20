import os
import json
from entities import GithubIssue
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import data_preprocessor
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import click
import random
import pandas as pd
from tqdm import tqdm
import pickle


dataset_name = 'tf_vuln_dataset.csv'

stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

directory = os.path.dirname(os.path.abspath(__file__))
github_issue_folder_path = os.path.join(directory, 'github_issues')
similarity_scores_file_path = os.path.join(directory, 'similarity_scores.txt')

source_code_extensions = ['.ios', '.c', '.java7', '.scala', '.cpp', '.php', '.cc', '.js', '.html',
                         '.swift', '.h', '.java', '.css', '.py']

c_notation_re = '[A-Za-z]+[0-9]*_.*'
qualified_name_re = '[A-Za-z]+[0-9]∗[\\.].+'
camel_case_re = '[A-Za-z]+.*[A-Z]+.*'
upper_case_re = '[A-Z0-9]+'
system_variable_re = '_+[A-Za-z0-9]+.+'
reference_expression_re = '[a-zA-Z]+[:]{2,}.+'

non_alphanumeric_pattern = re.compile(r'\W+', re.UNICODE)
hyper_link_pattern = re.compile(r"http\S+", re.UNICODE)
contain_both_number_and_char_pattern = re.compile(r'^(?=.*[a-zA-Z])(?=.*[0-9])', re.UNICODE)
regex = re.compile('|'.join([c_notation_re, qualified_name_re,
                             camel_case_re, upper_case_re,
                             system_variable_re, reference_expression_re]))

terms_min_length = 0

chunk_size = -1

all_file_extension = set()


def load_tensor_flow_issues():
    issues = []
    for file_name in os.listdir(github_issue_folder_path):
        if file_name.endswith('.json'):
            with open(github_issue_folder_path + '/' + file_name) as file:
                json_raw = file.read()
                json_dict_list = json.loads(json_raw)
                issues.extend(json_dict_list)
            # todo remove break
            # break
    return issues

def retrieve_code_terms(text):
    match_terms = []
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if not line.startswith("import") and not line.startswith("- import") and not line.startswith("+ import"):
            new_lines.append(line)

    text = " ".join(new_lines)
    tokens = word_tokenize(text)
    # tokens = text.split(" ")

    for token in tokens:
        if re.fullmatch(regex, token) and not token.isnumeric():
            # lowercase all token and split terms by '.' e.g, dog.speakNow -> ['dog', 'speakNow']
            parts = token.split('.')
            for part in parts:
                match_terms.extend(data_preprocessor.camel_case_split(part))

    match_terms = [token.lower() for token in data_preprocessor.under_score_case_split(match_terms)]
    match_terms = [token for token in match_terms if token not in stopwords_set]
    match_terms = [stemmer.stem(token) for token in match_terms]

    return match_terms


def extract_commit_code_terms(message, patch_list):
    terms = []
    retrieve_code_terms(message)
    for patch in patch_list:
        terms.extend(retrieve_code_terms(patch))

    return " ".join(terms)


def extract_issue_code_terms(issue):
    terms = []

    if issue['title'] is not None:
        terms.extend(retrieve_code_terms(issue['title']))

    if issue['body'] is not None:
        terms.extend(retrieve_code_terms(issue['body']))

    for comment in issue['comments']:
        terms.extend(retrieve_code_terms(comment))

    return " ".join(terms)


def extract_text(text):
    # filter hyperlinks
    # remove non-sense lengthy token, e.g 13f79535-47bb-0310-9956-ffa450edef68
    # remove numeric token
    # remove token contains both number(s) and character(s), e.g ffa450edef68
    raw_tokens = [token for token in text.split(' ') if not re.fullmatch(hyper_link_pattern, token)
                  # and not re.fullmatch(regex, token)
                  and not token.isnumeric()
                  and len(token) < 20
                  and not re.fullmatch(contain_both_number_and_char_pattern, token)]

    text = " ".join(raw_tokens)

    text = non_alphanumeric_pattern.sub(' ', text)

    tokens = word_tokenize(text)

    code_terms = retrieve_code_terms(text)
    tokens.extend(code_terms)

    tokens = [token for token in tokens if not re.fullmatch(regex, token)
              and not token.isnumeric()
              and len(token) < 20
              and not re.fullmatch(contain_both_number_and_char_pattern, token)]


    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token not in stopwords_set]

    tokens = [stemmer.stem(token) for token in tokens]

    if len(tokens) < terms_min_length:
        return []

    if chunk_size == -1:
        return [" ".join(tokens)]

    parts = []
    index = 0
    while index < len(tokens):
        if len(tokens) - index < terms_min_length:
            break

        parts.append(" ".join(tokens[index:min(index + chunk_size, len(tokens))]))
        index += chunk_size

    return parts


def is_non_source_document(file_name):
    for extension in source_code_extensions:
        if file_name.endswith(extension):
            return False

    return True


def extract_commit_text_terms_parts(message):
    terms_parts = []

    text_term = extract_text(message)
    if len(text_term) > 0:
        terms_parts = text_term

    # # tensorflow dataset does not have non-source document
    # for file in record.commit.files:
    #     if is_non_source_document(file.file_name) and file.patch is not None:
    #         text_term = extract_text(file.patch)
    #         if text_term is not None:
    #             terms_parts.extend(text_term)

    return terms_parts


def extract_issue_text_terms_parts(issue):
    terms_parts = []

    if issue['title'] is not None:
        text_term = extract_text(issue['title'])
        if len(text_term) > 0:
            terms_parts.extend(text_term)

    if issue['body'] is not None:
        text_term = extract_text(issue['body'])
        if len(text_term) > 0:
            terms_parts.extend(text_term)

    for comment in issue['comments']:
        text_term = extract_text(comment)
        if len(text_term) > 0:
            terms_parts.extend(text_term)

    return terms_parts


def get_tfidf_for_words(tfidf_matrix, feature_names, corpus_index):
    # get tfidf values from matrix instead of transform text => save time
    feature_index = tfidf_matrix[corpus_index, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[corpus_index, x] for x in feature_index])
    score_dict = {}
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        score_dict[w] = s
    return score_dict


def calculate_similarity(record_term_scores, ticket_term_scores):
    term_set = set()

    if len(record_term_scores) == 0 or len(ticket_term_scores) == 0:
        return 0

    if len(set(record_term_scores.keys()) & set(ticket_term_scores.keys())) == 0:
        return 0

    for term, value in record_term_scores.items():
        term_set.add(term)
    for term, value in ticket_term_scores.items():
        term_set.add(term)

    term_to_record_score = {}
    term_to_ticket_score = {}

    for term in term_set:
        if term in record_term_scores:
            term_to_record_score[term] = record_term_scores[term]
        if term in ticket_term_scores:
            term_to_ticket_score[term] = ticket_term_scores[term]

    # calculate cosine similarity
    numerator = 0
    for term in term_set:
        if term in record_term_scores and term in ticket_term_scores:
            numerator += term_to_record_score[term]*term_to_ticket_score[term]

    sub1 = 0
    for term, value in term_to_record_score.items():
        sub1 += value ** 2
    sub1 = math.sqrt(sub1)

    sub2 = 0
    for term, value in term_to_ticket_score.items():
        sub2 += value ** 2
    sub2 = math.sqrt(sub2)

    denominator = sub1 * sub2
    score = numerator / denominator
    return score


def link_similarity(record, issue, corpus_id_to_tf_idf_score,
                    url_to_corpus_id, issue_to_corpus_id):
    record_corpus_ids = sorted(url_to_corpus_id[record[0]])
    issue_corpus_ids = sorted(issue_to_corpus_id[issue['number']])

    # first documents are code terms documents
    max_score = 0

    # if len(corpus_id_to_tf_idf_score[record_corpus_ids[0]]) >= terms_min_length \
    #         and len(corpus_id_to_tf_idf_score[issue_corpus_ids[0]]) >= terms_min_length:
    max_score = calculate_similarity(corpus_id_to_tf_idf_score[record_corpus_ids[0]],
                                        corpus_id_to_tf_idf_score[issue_corpus_ids[0]])
    
    # if max_score == 0:
    #     return 0

    # calculate text terms similarity scores
    for record_document_id in record_corpus_ids[1:]:
        if len(corpus_id_to_tf_idf_score[record_document_id]) < terms_min_length:
            continue

        for issue_document_id in issue_corpus_ids[1:]:
            if len(corpus_id_to_tf_idf_score[issue_document_id]) < terms_min_length:
                continue

            max_score = max(max_score, calculate_similarity(corpus_id_to_tf_idf_score[record_document_id],
                                                            corpus_id_to_tf_idf_score[issue_document_id]))
    return max_score


def calculate_corpus_document_score(tfidf_matrix, feature_names, corpus):
    id_to_score = {}
    for index in tqdm(range(len(corpus))):
        id_to_score[index] = get_tfidf_for_words(tfidf_matrix, feature_names, index)

    return id_to_score


def calculate_vectorizer(records, issues, tfidf_vectorizer, url_to_code_terms, url_to_text_terms, issue_to_code_terms, issue_to_text_terms):
    corpus = []
    corpus_index = -1

    url_to_corpus_id = {}
    for record in records:
        corpus_index += 1
        url = record[0]
        url_to_corpus_id[url] = [corpus_index]
        corpus.append(url_to_code_terms[url])

        for text_terms in url_to_text_terms[url]:
            corpus_index += 1
            url_to_corpus_id[url].append(corpus_index)
            corpus.append(text_terms)

    issue_to_corpus_id = {}
    for issue in issues:
        corpus_index += 1
        issue_number = issue['number']
        issue_to_corpus_id[issue_number] = [corpus_index]
        corpus.append(issue_to_code_terms[issue_number])
      
        for text_terms in issue_to_text_terms[issue_number]:
            corpus_index += 1
            issue_to_corpus_id[issue_number].append(corpus_index)
            corpus.append(text_terms)

    print("Calculating TF-IDF vectorizer...")
    # tfidf_vectorizer.fit(corpus)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names()
    print("Finish calculating TF-IDF vectorizer")

    # print("Start calculating TF-IDF score for every words in every document in corpus...")
    # corpus_id_to_tfidf_score = calculate_corpus_document_score(tfidf_matrix, feature_names, corpus)
    # print("Finish calculating TF-IDF score")

    with open("tf_issue_linker_tf_idf_vectorizer.pickle", 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    with open("tf_issue_linker_tf_idf_maxtrix.pickle", 'wb') as file:
        pickle.dump(tfidf_matrix, file)
        

def calculate_similarity_scores(records, issues, tfidf_vectorizer, url_to_code_terms, url_to_text_terms, issue_to_code_terms, issue_to_text_terms):
    corpus = []
    corpus_index = -1

    url_to_corpus_id = {}
    for record in records:
        corpus_index += 1
        url = record[0]
        url_to_corpus_id[url] = [corpus_index]
        corpus.append(url_to_code_terms[url])

        for text_terms in url_to_text_terms[url]:
            corpus_index += 1
            url_to_corpus_id[url].append(corpus_index)
            corpus.append(text_terms)

    issue_to_corpus_id = {}
    for issue in issues:
        corpus_index += 1
        issue_number = issue['number']
        issue_to_corpus_id[issue_number] = [corpus_index]
        corpus.append(issue_to_code_terms[issue_number])
      
        for text_terms in issue_to_text_terms[issue_number]:
            corpus_index += 1
            issue_to_corpus_id[issue_number].append(corpus_index)
            corpus.append(text_terms)

    print("Calculating TF-IDF vectorizer...")
    # tfidf_vectorizer.fit(corpus)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names()
    print("Finish calculating TF-IDF vectorizer")

    print("Start calculating TF-IDF score for every words in every document in corpus...")
    corpus_id_to_tfidf_score = calculate_corpus_document_score(tfidf_matrix, feature_names, corpus)
    print("Finish calculating TF-IDF score")

    score_lines = []
    record_count = 0
    for record in tqdm(records):
        url = record[0]
        max_score = 0
        best_ticket = None
        for issue in issues:
            current_score = link_similarity(record, issue, corpus_id_to_tfidf_score,
                                            url_to_corpus_id, issue_to_corpus_id)
            if current_score > max_score:
                max_score = current_score
                best_ticket = issue['number']
        if best_ticket is not None:
            score_lines.append((url, best_ticket, max_score))
            # score_lines.append(str(record.id) + '\t\t' + record.repo + '/commit/' + record.commit_id + '\t\t'
            #                    + str(best_ticket.id)
            #                    + '\t\t' + str(max_score) + '\t\t' + best_ticket.name)
        else:
            score_lines.append(url, -1, -1)
            # score_lines.append(
            #     str(record.id) + '\t\t' + record.repo + '/commit/' + record.commit_id + '\t\t' + 'None'
            #     + '\t\t' + '0' + '\t\t' + 'None')

        # if record_count % 50 == 0:
        #     print("Finished {} records".format(record_count))

    df = pd.DataFrame((list(score_lines)), columns=['url', 'ticket', 'score'])
    df.to_csv('tf_issue_linking.csv', index=False)


def load_tensor_flow_records():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition']]

    records = []

    items = df.to_numpy().tolist()

    url_to_message, url_to_diff= {}, {}


    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        
        message = item[2]
        diff = item[4]

        if pd.isnull(diff):   
            continue

        if url not in url_to_message:
            url_to_message[url] = message

        if url not in url_to_diff:
            url_to_diff[url] = []

        url_to_diff[url].append(diff)

    for url, message in url_to_message.items():
        diff = url_to_diff[url]
        records.append((url, message, diff))

    return records


def pre_calculate_issue_vectorizer():
    global terms_min_length
    terms_min_length = 0

    global chunk_size
    chunk_size = -1

    min_df = 3
    
    print("Setting:")
    print("     Text terms min length: {}".format(terms_min_length))

    records = load_tensor_flow_records()

    print("Records length: {}".format(len(records)))

    print("Start extract commit features...")
    short_term_count = 0
    url_to_code_terms = {}
    url_to_text_terms = {}
    for record in tqdm(records):
        
        url = record[0]
        code_terms = extract_commit_code_terms(record[1], record[2])
        # record.code_terms = ''
        text_terms_parts = extract_commit_text_terms_parts(record[1])

        url_to_code_terms[url] = code_terms
        url_to_text_terms[url] = text_terms_parts 

    print("Finish extract commit features")

    issues = load_tensor_flow_issues()

    # random.shuffle(jira_tickets)
    print("Issues length: {}".format(len(issues)))
    print("Start extracting issue features...")
    
    issue_to_code_terms = {}
    issue_to_text_terms = {}

    for issue in tqdm(issues):
        issue_number = issue['number']
        code_terms = extract_issue_code_terms(issue)
        # issue.code_terms = ''
        text_terms_parts = extract_issue_text_terms_parts(issue)

        issue_to_code_terms[issue_number] = code_terms
        issue_to_text_terms[issue_number] = text_terms_parts

    print("Finish extracting issue features")
    tfidf_vectorizer = TfidfVectorizer()
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    # if max_df != 1:
    #     tfidf_vectorizer.max_df = max_df

    calculate_vectorizer(records, issues, tfidf_vectorizer, url_to_code_terms, url_to_text_terms, issue_to_code_terms, issue_to_text_terms)


def process_linking():

    # test_true_link is option for testing how many percent of records in our dataset link to their real issues
    # merge_link is option to choose whether we merge "real issues" to "crawled issues" to check the ability of
    # issue linker to recover true link

    # global terms_min_length
    # terms_min_length = text_feature_min_length

    # global similarity_scores_file_path
    # similarity_scores_file_path = os.path.join(directory, output_file_name)

    # global chunk_size
    # chunk_size = chunk


    global terms_min_length
    terms_min_length = 0

    global chunk_size
    chunk_size = -1

    min_df = 3
    
    print("Setting:")
    print("     Text terms min length: {}".format(terms_min_length))

    records = load_tensor_flow_records()

    print("Records length: {}".format(len(records)))

    print("Start extract commit features...")
    short_term_count = 0
    url_to_code_terms = {}
    url_to_text_terms = {}
    for record in tqdm(records):
        
        url = record[0]
        code_terms = extract_commit_code_terms(record[1], record[2])
        # record.code_terms = ''
        text_terms_parts = extract_commit_text_terms_parts(record[1])

        url_to_code_terms[url] = code_terms
        url_to_text_terms[url] = text_terms_parts 

        # need_print = False
        # for terms in text_terms_parts:
        #     if len(terms) <= 10:
        #         need_print = True

        # if not need_print:
        #     continue
        # short_term_count += 1

    print("Finish extract commit features")

    issues = load_tensor_flow_issues()

    # random.shuffle(jira_tickets)
    print("Issues length: {}".format(len(issues)))
    print("Start extracting issue features...")
    
    issue_to_code_terms = {}
    issue_to_text_terms = {}

    for issue in tqdm(issues):
        issue_number = issue['number']
        code_terms = extract_issue_code_terms(issue)
        # issue.code_terms = ''
        text_terms_parts = extract_issue_text_terms_parts(issue)

        issue_to_code_terms[issue_number] = code_terms
        issue_to_text_terms[issue_number] = text_terms_parts

    print("Finish extracting issue features")
    tfidf_vectorizer = TfidfVectorizer()
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    # if max_df != 1:
    #     tfidf_vectorizer.max_df = max_df

    calculate_similarity_scores(records, issues, tfidf_vectorizer, url_to_code_terms, url_to_text_terms, issue_to_code_terms, issue_to_text_terms)

def get_score_for_single_document(vectorizer, document):
    tf_idf = vectorizer.transform([document])
    feature_names = vectorizer.get_feature_names()
    feature_index = tf_idf[0, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tf_idf[0, x] for x in feature_index])
    score_dict = {}
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        score_dict[w] = s
    return score_dict

def infer_issue(url, message, diffs):
    # with open("tf_issue_linker_vectorizer.pickle", 'rb') as file:
    #     tfidf_matrix = pickle.load(file)

    with open('tf_issue_linker_corpus_score.dict', 'rb') as file:
        corpus_id_to_tfidf_score = pickle.load(file)

    # with open('tf_issue_linker_tf_idf_maxtrix.pickle', 'rb') as file:
    #     tfidf_matrix = pickle.load(file)

    with open('tf_issue_linker_tf_idf_vectorizer.pickle', 'rb') as file:
        vectorizer = pickle.load(file)
 
    # extract commit's terms
    record = (url, message, diffs)

    code_terms = extract_commit_code_terms(message, diffs)
    text_terms_parts = extract_commit_text_terms_parts(message)
    url_to_code_terms = {url:code_terms}
    url_to_text_terms = {url:text_terms_parts}
    url_to_corpus_id = {}


    # need to add issues terms to corpus first to sync with pre-calculated score
    corpus_index = -1
    corpus = []

    # loading issues

    issues = load_tensor_flow_issues()

    issue_to_code_terms, issue_to_text_terms = load_issue_terms()
    
    issue_to_corpus_id = {}
    for issue in issues:
        corpus_index += 1
        issue_number = issue['number']
        issue_to_corpus_id[issue_number] = [corpus_index]
        corpus.append(issue_to_code_terms[issue_number])
      
        for text_terms in issue_to_text_terms[issue_number]:
            corpus_index += 1
            issue_to_corpus_id[issue_number].append(corpus_index)
            corpus.append(text_terms)

    # add record to corpus later
    corpus_index += 1
    url_to_corpus_id[url] = [corpus_index]
    corpus.append(url_to_code_terms[url])

    for text_terms in url_to_text_terms[url]:
        corpus_index += 1
        url_to_corpus_id[url].append(corpus_index)
        corpus.append(text_terms)

    for corpus_id in url_to_corpus_id[url]:
        corpus_id_to_tfidf_score[corpus_id] = get_score_for_single_document(vectorizer, corpus[corpus_id])
    # find best issue

    max_score = 0
    best_issue = issue
    for issue in issues:
        current_score = link_similarity(record, issue, corpus_id_to_tfidf_score,
                                        url_to_corpus_id, issue_to_corpus_id)
        if current_score > max_score:
            max_score = current_score
            best_issue = issue

    # print("Best issue: {}".format(best_issue['number']))
    # print(max_score)
    # print(best_issue['title'])
    # print(best_issue['body'])
    return best_issue    


def extract_issue_terms():
    issues = load_tensor_flow_issues()

    issue_to_code_terms = {}
    issue_to_text_terms = {}

    for issue in tqdm(issues):
        issue_number = issue['number']
        code_terms = extract_issue_code_terms(issue)
        # issue.code_terms = ''
        text_terms_parts = extract_issue_text_terms_parts(issue)

        issue_to_code_terms[issue_number] = code_terms
        issue_to_text_terms[issue_number] = text_terms_parts

    with open('issue_to_code_terms.dict', 'wb') as file:
        pickle.dump(issue_to_code_terms, file)
    
    with open('issue_to_text_terms.dict', 'wb') as file:
        pickle.dump(issue_to_text_terms, file)
    

def load_issue_terms():
    with open('issue_to_code_terms.dict', 'rb') as file:
        issue_to_code_terms = pickle.load(file)
    
    with open('issue_to_text_terms.dict', 'rb') as file:
        issue_to_text_terms = pickle.load(file)

    return issue_to_code_terms, issue_to_text_terms


def pre_calculate_document_score():
    issues = load_tensor_flow_issues()
    issue_to_code_terms, issue_to_text_terms = load_issue_terms()

    corpus = []
    corpus_index = -1
    issue_to_corpus_id = {}
    for issue in issues:
        corpus_index += 1
        issue_number = issue['number']
        issue_to_corpus_id[issue_number] = [corpus_index]
        corpus.append(issue_to_code_terms[issue_number])
      
        for text_terms in issue_to_text_terms[issue_number]:
            corpus_index += 1
            issue_to_corpus_id[issue_number].append(corpus_index)
            corpus.append(text_terms)
    
    print("Loading TF-IDF vectorizer...")
    # tfidf_vectorizer.fit(corpus)
    with open('tf_issue_linker_matrix.pickle', 'rb') as file:
        tfidf_matrix = pickle.load(file)

    with open('tf_issue_linker_vectorizer.pickle', 'rb') as file:
        vectorizer = pickle.load(file)
        feature_names = vectorizer.get_feature_names()

    print("Finish loading TF-IDF vectorizer")

    print("Start calculating TF-IDF score for every words in every document in corpus...")
    corpus_id_to_tfidf_score = calculate_corpus_document_score(tfidf_matrix, feature_names, corpus)
    print("Finish calculating TF-IDF score")

    with open('tf_issue_linker_corpus_score.dict', 'wb') as file:
        pickle.dump(corpus_id_to_tfidf_score, file)


if __name__ == '__main__':
    # print("Pre-calculating data...")
    # pre_calculate_issue_vectorizer()
    # print("Finish calculating")

    # print("Extracting and saving issue terms...")
    # extract_issue_terms()
    # print("Finish extracting and saving")

    # pre_calculate_document_score()

    url = 'abc.xyz'
    message = "Prevent memory leak in decoding PNG images. PiperOrigin-RevId: 409300653 Change-Id: I6182124c545989cef80cefd439b659095920763b"
    patch = ["@@ -78,11 +78,24 @@ class SparseDenseBinaryOpShared : public OpKernel {\n                     \"but received shapes: \",\n                     values_t->shape().DebugString(), \" and \",\n                     shape_t->shape().DebugString()));\n+    OP_REQUIRES(\n+        ctx, TensorShapeUtils::IsVector(shape_t->shape()),\n+        errors::InvalidArgument(\"Input sp_shape must be a vector. Got: \",\n+                                shape_t->shape().DebugString()));\n     OP_REQUIRES(\n         ctx, values_t->dim_size(0) == indices_t->dim_size(0),\n         errors::InvalidArgument(\n             \"The first dimension of values and indices should match. (\",\n             values_t->dim_size(0), \" vs. \", indices_t->dim_size(0), \")\"));\n+    OP_REQUIRES(\n+        ctx, shape_t->shape().dim_size(0) == indices_t->shape().dim_size(1),\n+        errors::InvalidArgument(\n+            \"Number of dimensions must match second dimension of indices. \",\n+            \"Got \", shape_t->shape().dim_size(0),\n+            \" dimensions, indices shape: \", indices_t->shape().DebugString()));\n+    OP_REQUIRES(ctx, shape_t->NumElements() > 0,\n+                errors::InvalidArgument(\n+                    \"The shape argument requires at least one element.\"));\n \n     const auto indices_mat = indices_t->matrix<int64_t>();\n     const auto shape_vec = shape_t->vec<int64_t>();\n"]     
    infer_issue(url, message, patch)
