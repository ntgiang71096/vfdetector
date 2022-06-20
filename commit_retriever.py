import pydriller
from pydriller import Repository, ModificationType
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

directory = os.path.dirname(os.path.abspath(__file__))
repo_folder = '/Users/nguyentruonggiang/PycharmProjects/tensorflow'


def is_valid_filename(filename):
    return filename.endswith(('.cc', '.h', '.py', '.c'))


def retrieve_commit_data(src_file, column_name, has_prefix, dest_file, label):
    df = pd.read_csv(src_file)
    url_list = df[column_name].values.tolist()

    # retrieve line < 200
    # .cc, .h, .py, .c
    # outliers:
    # 1970c2158b1ffa416d159d03c3370b9a462aee35
    # e11f55585f614645b360563072ffeb5c3eeff162
    lines = []
    items = []
    for url in tqdm(url_list):
        sha = url
        if has_prefix:
            sha = url[len('https://github.com/tensorflow/tensorflow/commit/'):]

        has_file = False
        for commit in Repository(repo_folder, single=sha).traverse_commits():
            # lines.append(commit.lines)
            # if commit.lines > 300:
            #     print(sha)
            # for file in commit.modified_files:
            #     print(file.filename)

            msg = commit.msg
            for file in commit.modified_files:
                if is_valid_filename(file.filename):
                    items.append((sha, 'tensorflow/tensorflow', msg, file.filename, file.diff, label))
                    has_file = True
        if not has_file:
            print(url)

    df = pd.DataFrame(items, columns=['commit_id', 'repo', 'msg', 'filename', 'diff', 'label'])
    df.to_csv(dest_file, index=False)

    # lines = sorted(lines)
    # plt.figure()
    # plt.boxplot(lines, labels=['lines'], showfliers=False)
    # plt.title('lines distribution')
    # plt.ylabel('Highest weight')
    # plt.xlabel('Segment')
    # plt.show()
    # print()


def retrieve_candidate_neg_commit():
    df = pd.read_csv('tf_fixes.csv')
    url_list = df['commit_url'].values.tolist()
    pos_sha_list = [url[len('https://github.com/tensorflow/tensorflow/commit/'):] for url in url_list]

    all_sha_list = set()

    for commit in tqdm(Repository(repo_folder).traverse_commits()):
        if commit not in pos_sha_list:
            all_sha_list.add(commit.hash)
            print(commit.hash)

    df = pd.DataFrame((list(all_sha_list)), columns=['neg_sha'])
    df.to_csv('neg_candidate.csv', index=False)


def has_valid_file(commit):
    for file in commit.modified_files:
        if is_valid_filename(file.filename) and \
                file.change_type in [ModificationType.ADD, ModificationType.MODIFY, ModificationType.DELETE]:
            return True

    return False


def retrieve_neg_commit():
    total = 307 * 5

    candidate_list = pd.read_csv('neg_candidate.csv')['neg_sha'].values.tolist()
    chosen_list = set()
    neg_sha_list = []
    count = 0
    while count < total:
        sha = random.choice(candidate_list)
        print(sha)

        if sha in chosen_list:
            continue

        chosen_list.add(sha)

        for commit in Repository(repo_folder, single=sha).traverse_commits():
            if commit.lines > 200:
                # remove outliers
                continue
            if not has_valid_file(commit):
                continue

            count += 1
            neg_sha_list.append(sha)
            print("Selected")

    df = pd.DataFrame((list(neg_sha_list)), columns=['selected_neg_sha'])
    df.to_csv('selected_neg_sha.csv', index=False)


def partition_dataset():
    pos_ids = list(set(pd.read_csv('tf_pos.csv')['commit_id'].values.tolist()))
    neg_ids = list(set(pd.read_csv('tf_neg.csv')['commit_id'].values.tolist()))

    pos_train, pos_test, _, _ = train_test_split(pos_ids, [0] * len(pos_ids), test_size=0.2, random_state=109)
    neg_train, neg_test, _, _ = train_test_split(neg_ids, [0] * len(neg_ids), test_size=0.2, random_state=109)

    items = []

    pos_items = pd.read_csv('tf_pos.csv')[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label']].values.tolist()
    neg_items = pd.read_csv('tf_neg.csv')[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label']].values.tolist()

    for item in pos_items:
        if item[0] in pos_train:
            item.append('train')
        else:
            item.append('test')
        items.append(item)

    for item in neg_items:
        if item[0] in neg_train:
            item.append('train')
        else:
            item.append('test')
        items.append(item)

    df = pd.DataFrame(items, columns=['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition'])
    df.to_csv('tf_vuln_dataset.csv', index=False)


# retrieve_commit_data(src_file='tf_fixes.csv', column_name='commit_url',
#                      has_prefix=True, dest_file='tf_pos.csv', label=1)

partition_dataset()