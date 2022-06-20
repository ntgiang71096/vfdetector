from email.quoprimime import body_check
import pandas as pd
import data_loader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import issue_linker
from entities import Record, GithubIssue, GithubIssueComment, GithubCommit, GithubCommitFile, EntityEncoder
import json

sap_dataset_name = 'full_dataset_with_all_features.txt'

sap_new_dataset_name = 'sap_patch_dataset.csv'

def convert_sap_to_huawei_format():
    records = data_loader.load_records(sap_dataset_name)
    patches, labels, urls = [], [], []
    url_to_partition = {}

    for record in records:
        url = record.repo + '/commit/' + record.commit_id
        url = url[len('https://github.com/') :]
        # print(url)
        urls.append(url)

    url_train, url_test, _, _ = train_test_split(urls, [0] * len(urls), test_size=0.20, random_state=109)
    print(len(url_train))
    print(len(url_test))
    for url in url_train:
        url_to_partition[url] = 'train'

    for url in url_test:
        url_to_partition[url] = 'test'

    # item format ('commit_id', 'repo', 'partition', 'diff', 'label', 'PL')
    items = []

    for record in tqdm(records):
        url = record.repo + '/commit/' + record.commit_id
        url = url[len('https://github.com/') :]
        label = record.label
        files = record.commit.files
        repo = record.repo[len('https://github.com/') :]
        has_file = False
        for file in files:
            if file.file_name.endswith(('.java', '.py')) and file.patch is not None:
                has_file = True
                items.append((record.commit_id, repo, url_to_partition[url], file.patch, label, 'N/A' ))

        if not has_file:
            items.append((record.commit_id, repo, url_to_partition[url], "empty", label, 'N/A' ))
    df = pd.DataFrame(items, columns=['commit_id', 'repo', 'partition', 'diff', 'label', 'PL'])

    df.to_csv(sap_new_dataset_name, index=False)


def convert_to_sap_format():
    # dataset_name = 'tf_vuln_dataset.csv'
    records = issue_linker.load_tensor_flow_records()
    issues = issue_linker.load_tensor_flow_issues()
    number_to_issue = {}
    for issue in issues:
        number = issue['number']
        number_to_issue[number] = issue

    url_to_number = {}
    df = pd.read_csv('tf_issue_linking.csv')
    for item in df.values.tolist():
        url_to_number[item[0]] = item[1]

    sap_data = []
    for record in tqdm(records):
        url = record[0]
        msg = record[1]
        diffs = record[2]
        repo, commit_id = url.split('/commit/')

        sap_record = Record()
        sap_record.commit_message = msg
        sap_record.repo = repo
        sap_record.commit_id = commit_id

        github_commit = GithubCommit()
        commit_files = []

        for diff in diffs:
            file = GithubCommitFile()
            file.patch = diff
            commit_files.append(file)
        
        github_commit.files = commit_files
        sap_record.commit = github_commit
        
        issue_json = number_to_issue[url_to_number[url]]

        issue = GithubIssue()
        issue.title = issue_json['title']
        issue.body = issue_json['body']
        comments = []
        for comment_json in issue_json['comments']:
            comment = GithubIssueComment()
            comment.body = comment_json
            comments.append(comment)

        issue.comments = comments
        
        sap_record.add_github_ticket(issue)

        sap_data.append(sap_record)

    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(sap_data)

    with open('tf_dataset_sap_format.txt', 'w') as file:
        file.write(json_value)
    print("Finishing writing")



if __name__ == '__main__':
    convert_to_sap_format()  
