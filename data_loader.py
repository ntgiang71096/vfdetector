import os
from entities import *

full_data_with_features_file_path = '../MSR2019/experiment/dataset_clean_updated.txt'
record_file_path = '../MSR2019/experiment/dataset_clean.txt'
github_commit_file_path = '../data/github_commit'
github_issue_file_path = '../data/github_issue'
jira_ticket_file_path = '../data/jira_ticket'
record_full_file_path = '../data/record_full.txt'


def load_github_commit(id_to_record):
    record_id_list = []
    # some commit with very large patches need to be removed
    filtered_records = []
    for file_name in os.listdir(github_commit_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        record_id_list.append(record_id)

    print(len(record_id_list))

    for file_name in os.listdir(github_commit_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        with open(github_commit_file_path + '/' + file_name) as file:
            content = file.read()
            commit = GithubCommit(json_value=content)
            record = id_to_record[record_id]
            record.commit = commit

    for id in record_id_list:
        filtered_records.append(id_to_record[id])

    return filtered_records


def load_github_issue(id_to_record):
    for file_name in os.listdir(github_issue_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        with open(github_issue_file_path + '/' + file_name) as file:
            json_raw = file.read()
            json_dict_list = json.loads(json_raw)
            github_issues = []
            for json_dict in json_dict_list:
                github_issues.append(GithubIssue(json_value=json_dict))
        if record_id in id_to_record:
            id_to_record[record_id].github_issue_list = github_issues


def load_jira_ticket(id_to_record):
    for file_name in os.listdir(jira_ticket_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        with open(jira_ticket_file_path + '/' + file_name) as file:
            json_raw = file.read()
            json_dict_list = json.loads(json_raw)
            jira_tickets = []
            for json_dict in json_dict_list:
                jira_tickets.append(JiraTicket(json_value=json_dict))
        id_to_record[record_id].jira_ticket_list = jira_tickets


def write_full_data_to_file(file_path):
    print("Writing full records to file...")
    records = load_records(record_file_path)
    
    id_to_record = {}
    for record in records:
        if utils.is_not_large_commit(record.commit):
            id_to_record[record.id] = record

    load_github_issue(id_to_record)
    load_jira_ticket(id_to_record)

    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(records)

    with open(file_path, 'w') as file:
        file.write(json_value)
    print("Finishing writing")


def load_records(file_path):
    print("Start loading records...")

    records = []
    with open(file_path, 'r') as file:
        json_raw = file.read()
        json_dict_list = json.loads(json_raw)
        for json_dict in json_dict_list:
            records.append(Record(json_value=json.dumps(json_dict)))

    print("Finish loading records")

    for record in records:
        if record.label is not None:
            if record.label == 'pos':
                record.label = 1
            elif record.label == 'neg':
                record.label = 0

    for record in records:
        record.id = int(record.id)

    return records


def remove_duplicate(file_path, new_file_path):
    records = load_records(file_path)
    print(len(records))

    patch_to_record = {}
    filtered_id = set()

    for record in records:
        commit = record.commit
        patch = ''
        for file in commit.files:
            if file.patch is not None:
                patch = patch + file.patch

        if patch not in patch_to_record:
            patch_to_record[patch] = record.id
        else:
            filtered_id.add(record.id)

    records = [record for record in records if record.id not in filtered_id]

    print(len(records))

    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(records)

    with open(new_file_path, 'w') as file:
        file.write(json_value)
    print("Finishing writing")

def modify_data_set(file_path):
    print("Writing full records to file...")
    records = load_records(record_file_path)

    for record in records:
        if record.repo == 'https://github.com/blynkkk/blynk-server':
            print("fixed")
            record.repo = 'https://github.com/Peterkn2001/blynk-server'

    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(records)

    with open(file_path, 'w') as file:
        file.write(json_value)
    print("Finishing writing")