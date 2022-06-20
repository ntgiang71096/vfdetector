from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from re import finditer
from nltk.tokenize import word_tokenize
import os
import csv
import json
from datetime import datetime
from time import mktime
from entities import EntityEncoder, GithubCommit, GithubCommitFile
import data_loader

non_alphanumeric_pattern = re.compile(r'\W+', re.UNICODE)
stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer()
directory = os.path.dirname(os.path.abspath(__file__))
pr_folder_path = os.path.join(directory, "data/github_statistics/pull_request")
max_timestamp = 999999999999

def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def under_score_case_split(words):
    result = []
    for word in words:
        tokens = word.split("_")
        result.extend(tokens)
    return result


def process_patch(patch, options):
    lines = patch.splitlines()
    full_tokens = []
    for line in lines:
        # remove context line, save changed lines only
        if options.use_patch_context_lines or line.startswith('-') or line.startswith('+'):

            # remove non alphanumeric characters
            line = non_alphanumeric_pattern.sub(' ', line)

            # split lines to tokens
            tokens = line.split()

            # convert every camel, under_score words to list of single tokens
            tokens = [camel_case_split(token) for token in tokens]
            tokens = [under_score_case_split(token) for token in tokens]

            for single_tokens in tokens:
                full_tokens.extend(single_tokens)

    # to lower case
    full_tokens = [token.lower() for token in full_tokens]

    # remove stop words
    full_tokens = [token for token in full_tokens if token not in stopwords_set]

    # stem words
    full_tokens = [stemmer.stem(token) for token in full_tokens]

    return ' '.join(full_tokens)


def process_textual_information(text, options):
    # remove non alphanumeric characters
    text = non_alphanumeric_pattern.sub(' ', text)

    tokens = word_tokenize(text)
    # convert every camel, under_score words to list of single tokens
    tokens = [camel_case_split(token) for token in tokens]
    tokens = [under_score_case_split(token) for token in tokens]

    full_tokens = []
    for single_tokens in tokens:
        full_tokens.extend(single_tokens)

    # to lower case
    full_tokens = [token.lower() for token in full_tokens]

    # remove stop words
    full_tokens = [token for token in full_tokens if token not in stopwords_set]

    # stem words
    full_tokens = [stemmer.stem(token) for token in full_tokens]

    if options.ignore_number:
        full_tokens = [token for token in full_tokens if not token.isdigit()]

    return ' '.join(full_tokens)


def attach_github_issue(info, github_issue_list, options):
    for github_issue in github_issue_list:
        if github_issue.title is not None:
            info = info + '\n' + github_issue.title
        if github_issue.body is not None:
            info = info + '\n' + github_issue.body

        if options.use_comments:
            for comment in github_issue.comments:
                info = info + '\n' + comment.body

    return info


def attach_jira_ticket(info, jira_ticket_list, options):
    for jira_ticket in jira_ticket_list:
        if jira_ticket.name is not None:
            info = info + '\n' + jira_ticket.name
        if jira_ticket.description is not None:
            info = info + '\n' + jira_ticket.description
        if jira_ticket.summary is not None:
            info = info + '\n' + jira_ticket.summary

        if options.use_comments:
            for comment in jira_ticket.comments:
                info = info + '\n' + comment.body

    return info


def preprocess_patch(record):
    # preprocess commit patches
    for commit_file in record.commit.files:
        if commit_file.patch is not None:
            commit_file.patch = process_patch(commit_file.patch)

    return record


def preprocess_single_record(record, options):
    # print("process single records")
    # preprocess commit patches
    # record = preprocess_patch(record)
    for commit_file in record.commit.files:
        if commit_file.patch is not None:
            pass
            # commit_file.patch = process_patch(commit_file.patch, options)

    # preprocess commit message
    # print(record)
    if record.commit_message is not None:
        pass
        # record.commit_message = process_textual_information(record.commit_message, options)

    record.issue_info = ''
    if options.use_issue_classifier:
        if len(record.github_issue_list) > 0:
            record.issue_info = attach_github_issue(record.issue_info, record.github_issue_list, options)
        if len(record.jira_ticket_list) > 0:
            record.issue_info = attach_jira_ticket(record.issue_info, record.jira_ticket_list, options)

    record.issue_info = ' '.join([token for token in record.issue_info.split(' ')[:500]])
    if record.issue_info != '':
        pass
        # record.issue_info = process_textual_information(record.issue_info, options)

    return record


def get_commit_id_to_repo(records):
    commit_id_to_repo = {}

    for record in records:
        repo = record.repo
        commit_id_to_repo[int(record.repo)] = repo

    return commit_id_to_repo


def get_pr_data(records, repo_to_username):
    repo_to_open_pr = {}
    repo_to_close_pr = {}
    repo_to_contributor_open_pr = {}
    repo_to_contributor_close_pr = {}

    commit_id_to_repo = get_commit_id_to_repo(records)

    for file_name in os.listdir(pr_folder_path):
        file_path = pr_folder_path + '/' + file_name
        commit_id = int(file_path.split('.')[0])
        repo = commit_id_to_repo[commit_id]
        repo_to_open_pr[repo] = 0
        repo_to_close_pr[repo] = 0
        repo_to_contributor_open_pr[repo] = {}
        repo_to_contributor_close_pr[repo] = {}

        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count == 0:
                    continue
                state = row[4]
                if state == 'open':
                    repo_to_open_pr[repo] += 1
                if state == 'closed':
                    repo_to_close_pr[repo] += 1
                username = row[5]

                if username in repo_to_username[repo]:
                    if username not in repo_to_contributor_open_pr[repo]:
                        repo_to_contributor_open_pr[repo][username] = 0
                    if username not in repo_to_contributor_close_pr[repo]:
                        repo_to_contributor_close_pr[repo][username] = 0

                    if state == 'open':
                        repo_to_contributor_open_pr[repo][username] += 1
                    if state == 'closed':
                        repo_to_contributor_close_pr[repo][username] += 1

    return repo_to_open_pr, repo_to_close_pr, repo_to_contributor_open_pr, repo_to_contributor_close_pr


def get_commit_timestamp(record):
    commit = record.commit
    date_string = commit.created_date
    date = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S GMT")
    return mktime(date.timetuple())


def get_timestamp(date_string):
    date = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S GMT")
    return mktime(date.timetuple())


def get_commit_count_before(data, commit_timestamp):
    username = data['author']['login']
    commit_count = 0
    for week_data in data['weeks']:
        if week_data['w'] >= commit_timestamp:
            continue
        commit_count += week_data['c']

    return commit_count


def get_repo_to_contributor_rank(repo_to_contributor_total_count):
    repo_to_contributor_rank = {}
    for repo in repo_to_contributor_total_count:
        repo_to_contributor_rank[repo] = {}
        contributor_total_count \
            = [(contributor, repo_to_contributor_total_count[repo][contributor])
               for contributor in repo_to_contributor_total_count[repo]]
        contributor_total_count.sort(key=lambda x: x[1], reverse=True)
        for index in range(len(contributor_total_count)):
            contributor = contributor_total_count[index][0]
            repo_to_contributor_rank[repo][contributor] = index + 1

    return repo_to_contributor_rank


def retrieve_contributor_fist_commit_time(contributor_data):
    min_date = max_timestamp
    weeks_data = contributor_data['weeks']

    for commit_data in weeks_data:
        current_date = commit_data['w']
        has_commit = commit_data['c'] > 0
        if has_commit and current_date < min_date:
            min_date = current_date

    return min_date


def retrieve_relevant_commit_count(record, record_to_contributor_commit_count):
    record_id = int(record.id)
    if record_id not in record_to_contributor_commit_count:
        return 0
    commit_count = record_to_contributor_commit_count[record_id]
    # if commit_count > 50:
    #     return 0
    return commit_count


def retrieve_relevant_latest_commit_time(record, record_to_contributor_latest_commit):
    default_date = datetime(day=1, month=1, year=1990)
    record_id = int(record.id)

    if record_id not in record_to_contributor_latest_commit:
        return mktime(default_date.timetuple())

    timestamp = record_to_contributor_latest_commit[record_id]
    return timestamp


def retrieve_relevant_first_commit_time(record, record_to_contributor_latest_commit):
    default_date = datetime(day=1, month=1, year=2030)
    record_id = int(record.id)

    if record_id not in record_to_contributor_latest_commit:
        return mktime(default_date.timetuple())

    timestamp = record_to_contributor_latest_commit[record_id]
    return timestamp


def get_record_to_contributor_commit_info(records, commit_to_username):
    # commit_count before the current commit date

    record_to_file_history = {}
    record_to_contributor_commit_count = {}
    record_to_contributor_latest_commit = {}
    record_to_contributor_first_commit = {}
    record_to_contributor_relevant_percentage = {}
    record_to_relevant_all_commit_count = {}

    file_history_folder_path = os.path.join(directory, 'data/github_statistics/file_history')
    for record in records:
        record_to_file_history[int(record.id)] = []

    print("Loading patches history...")
    for file_name in os.listdir(file_history_folder_path):
        if not file_name.endswith('.json'):
            continue

        parts = file_name.split('.')[0].split('_')
        record_id = int(parts[0])

        # in case test with small dataset
        if record_id not in record_to_file_history:
            continue

        with open(file_history_folder_path + '/' + file_name, 'r') as file:
            data = file.read()
        record_to_file_history[record_id].append(json.loads(data))

    print("Finished loading patches history")

    for record in records:
        record_id = int(record.id)
        repo = record.repo
        commit_id = record.commit_id
        commit_message = record.commit_message
        if record_id not in commit_to_username:
            continue

        commit_date = datetime.strptime(record.commit.created_date, "%a, %d %b %Y %H:%M:%S GMT")
        recent_date = datetime(day=1, month=1, year=1990)
        first_date = datetime(day=1, month=1, year=2030)
        # recent_date = datetime.strptime('1970-01-01', '%y-%m-%d')
        username = commit_to_username[record_id]
        file_history_list = record_to_file_history[record_id]
        commit_sha_set = set()
        total_commit_sha_set = set()
        has_date = False
        total_commit = 0
        for file_history in file_history_list:
            for file_data in file_history:
                # ignore data error
                if type(file_data) != dict:
                    continue
                # file history without commiter's username is ignored
                if file_data['committer'] is None:
                    continue
                if 'login' not in file_data['committer']:
                    continue
                committer_username = file_data['committer']['login']


                history_commit_date \
                    = datetime.strptime(file_data['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")

                commit_sha = file_data['sha']
                message = file_data['commit']['message']
                if commit_sha != commit_id \
                        and history_commit_date < commit_date:
                    # if history_commit_date.day == commit_date.day \
                    #         and history_commit_date.month == commit_date.month \
                    #         and (history_commit_date.day == commit_date.day or history_commit_date.day == commit_date.day - 1):
                    #     # print(record_id)
                    #     # print(repo + "/commit/" + commit_id)
                    #     # print(commit_message)
                    #     # print(message)
                    #     continue
                    has_date = True
                    total_commit_sha_set.add(commit_sha)
                    if username != committer_username:
                        continue
                    commit_sha_set.add(commit_sha)
                    if history_commit_date > recent_date:
                        recent_date = history_commit_date
                    if history_commit_date < first_date:
                        first_date = history_commit_date
        if has_date:
            record_to_contributor_commit_count[record_id] = len(commit_sha_set)
            record_to_contributor_latest_commit[record_id] = mktime(recent_date.timetuple())
            record_to_contributor_first_commit[record_id] = mktime(first_date.timetuple())
            record_to_contributor_relevant_percentage[record_id] = len(commit_sha_set) / len(total_commit_sha_set)
        else:
            record_to_contributor_commit_count[record_id] = 0
            record_to_contributor_latest_commit[record_id] = mktime(recent_date.timetuple())
            record_to_contributor_first_commit[record_id] = mktime(first_date.timetuple())
            record_to_contributor_relevant_percentage[record_id] = 0

    return record_to_contributor_commit_count, record_to_contributor_latest_commit, record_to_contributor_first_commit, record_to_contributor_relevant_percentage


def retrieve_relevant_percentage_commit(record, record_to_contributor_relevant_percentage):
    record_id = int(record.id)
    if record_id not in record_to_contributor_relevant_percentage:
        return 0
    percentage = record_to_contributor_relevant_percentage[record_id]

    return percentage


def retrieve_contributor_first_commit_timestamp(record, commit_to_username, repo_to_first_commit_time):
    repo = record.repo
    username = commit_to_username[int(record.id)]

    if repo not in repo_to_first_commit_time:
        return max_timestamp

    if username not in repo_to_first_commit_time[repo]:
        return max_timestamp

    return repo_to_first_commit_time[repo][username]


def retrieve_commiter_total_count(record, commit_to_username, repo_to_contributor_total_count):
    repo = record.repo
    record_id = int(record.id)
    if record_id not in commit_to_username:
        return 0

    username = commit_to_username[record_id]
    if repo not in repo_to_contributor_total_count:
        return 0

    if username not in repo_to_contributor_total_count[repo]:
        return 0

    return repo_to_contributor_total_count[repo][username]


def retrieve_commiter_rank(record, commit_to_username, repo_to_contributor_rank):
    repo = record.repo
    if int(record.id) not in commit_to_username:
        return 200
    username = commit_to_username[int(record.id)]

    # in case test with small data
    if repo not in repo_to_contributor_rank:
        return 200

    if username not in repo_to_contributor_rank[repo]:
        return 200

    return repo_to_contributor_rank[repo][username]


def retrieve_commiter_percentage(record, commit_to_username, repo_to_contributor_total_count, repo_to_total_count):
    # todo check return 0 or -1 is better
    record_id = int(record.id)
    repo = record.repo
    contributor = commit_to_username[record_id]
    if contributor == 'None-':
        return 0
        # return -1
    # in case test with small dataset
    if repo not in repo_to_contributor_total_count:
        return 0
        # return -1

    if contributor not in repo_to_contributor_total_count[repo]:
        return 0
        # return -1

    repo_total_count = repo_to_total_count[repo]
    contributor_total_count = repo_to_contributor_total_count[repo][contributor]
    percentage = contributor_total_count / repo_total_count

    return percentage


def get_repo_to_contributor_first_commit_time(repo_to_contributor_data):
    repo_to_first_commit_time = {}
    repo_to_repo_first_commit_time = {}
    for repo in repo_to_contributor_data:
        repo_to_first_commit_time[repo] = {}
        min_time = max_timestamp
        for username, contributor_data in repo_to_contributor_data[repo].items():
            time = retrieve_contributor_fist_commit_time(contributor_data)
            if time != max_timestamp:
                repo_to_first_commit_time[repo][username] = time
                min_time = min(min_time, time)
        if min_time != max_timestamp:
            repo_to_repo_first_commit_time[repo] = min_time

    return repo_to_first_commit_time, repo_to_repo_first_commit_time


def get_username_to_user_info():
    user_folder_path = os.path.join(directory, 'data/github_statistics/user')
    username_to_data = {}
    username_to_follower, username_to_number_of_repo, username_to_created_time = {}, {}, {}
    for file_name in os.listdir(user_folder_path):
        username = file_name.split(".")[0]
        with open(user_folder_path + '/' + file_name, 'r') as reader:
            data = json.loads(reader.read())
            username_to_data[username] = data

    # 2010-11-02T07:14:38Z
    for username, data in username_to_data.items():
        followers = data['followers']
        followers = min(followers, 1000)
        username_to_follower[username] = followers
        username_to_number_of_repo[username] = data['public_repos']

        date_string = data['created_at']
        date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
        timestamp = mktime(date.timetuple())
        username_to_created_time[username] = timestamp

    return username_to_follower, username_to_number_of_repo, username_to_created_time


def get_commit_to_username():
    commit_to_username = {}
    with open(os.path.join(directory, 'data/github_statistics/commit_username_new.csv'), mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            commit_id = int(row[0])
            username = row[3]
            commit_to_username[commit_id] = username
    return commit_to_username


def get_repo_to_contributor_data(id_to_record):
    contributor_activity_file_path = os.path.join(directory,
                                                  'data/github_statistics/contributor_activity')

    repo_to_total_count = {}
    repo_to_contributor_data = {}
    repo_to_contributor_total_count = {}
    for file_name in os.listdir(contributor_activity_file_path):
        if not file_name.endswith('.json'):
            continue
        record_id = int(file_name.split('.')[0])

        # in case test with small dataset
        if record_id not in id_to_record:
            continue
        repo = id_to_record[record_id].repo
        repo_to_total_count[repo] = 0
        repo_to_contributor_data[repo] = {}
        repo_to_contributor_total_count[repo] = {}
        with open(contributor_activity_file_path + '/' + file_name, 'r') as reader:
            item_list = json.loads(reader.read())
            for item in item_list:
                repo_to_total_count[repo] += item['total']
                username = item['author']['login']
                repo_to_contributor_data[repo][username] = item

                repo_to_contributor_total_count[repo][username] = item['total']

    return repo_to_contributor_data, repo_to_total_count, repo_to_contributor_total_count


def is_owner(commit_timestamp, committer, file_name, commit_list):
    committer_to_changes = {}
    for commit in commit_list:
        timestamp = get_timestamp(commit.created_date)
        if timestamp >= commit_timestamp:
            continue
        username = commit.author_name
        if username not in committer_to_changes:
            committer_to_changes[username] = 0
        for file in commit.files:
            current_file_name = file.file_name
            if current_file_name != file_name:
                continue
            committer_to_changes[username] += file.changes

    if len(committer_to_changes) == 0:
        return False

    committer_changes_list = []
    for username, changes in committer_to_changes.items():
        committer_changes_list.append((username, changes))

    committer_changes_list.sort(key=lambda x: x[1], reverse=True)

    if committer_changes_list[0][1] == 0:
        return False

    if committer_changes_list[0][0] == committer:
        return True

    return False


def retrieve_record_to_ownership(records, commit_to_username):
    # file_history_folder_path = os.path.join(directory, 'data/github_statistics/file_history')
    file_statistics_folder_path = os.path.join(directory, 'data/github_statistics/history_fast')
    repo_to_commit_statistics = {}
    # entity_encoder = EntityEncoder()
    repo_to_file_commits = {}

    record_id_to_repo = {}
    for record in records:
        record_id = int(record.id)
        record_id_to_repo[record_id] = record.repo

    print("Loading all related commits...")
    for file_name in os.listdir(file_statistics_folder_path):
        file_path = file_statistics_folder_path + '/' + file_name
        record_id = int(file_name.split('.')[0])
        repo = record_id_to_repo[record_id]
        repo_to_file_commits[repo] = {}
        with open(file_path, 'r') as reader:
            json_raw = reader.read()
            json_dict_list = json.loads(json_raw)
            for json_dict in json_dict_list:
                commit = GithubCommit(json_value=json.dumps(json_dict))
                for file in commit.files:
                    file_name = file.file_name
                    if file_name not in repo_to_file_commits[repo]:
                        repo_to_file_commits[repo][file_name] = []
                    repo_to_file_commits[repo][file_name].append(commit)

    print("Finish loading all commits")
    record_to_ownership = {}

    print("Calculating ownership...")
    for record in records:
        repo = record.repo
        record_id = int(record.id)
        commit_timestamp = get_commit_timestamp(record)
        # commit_list = repo_to_commit_statistics[repo]
        commit = record.commit

        # type 0: no valid file or username
        # type 1: owner
        # type 2: non owner
        # type 3: mix

        owner_type = 0
        owner = False
        non_owner = False
        if record_id not in commit_to_username:
            record_to_ownership[record_id] = 0

        username = commit_to_username[record_id]
        if username == 'None-':
            record_to_ownership[record_id] = 0
            continue

        for file in commit.files:
            file_name = file.file_name

            if 'src/test' in file_name:
                continue
            if not file_name.endswith(('.java', '.c', '.h')):
                continue

            # file name not present
            if file_name not in repo_to_file_commits[repo]:
                print(file_name)
                record_to_ownership[record_id] = 0
                continue

            if is_owner(commit_timestamp, username, file_name, repo_to_file_commits[repo][file_name]):
                owner = True
            else:
                non_owner = True
        if owner and not non_owner:
            owner_type = 1
        if not owner and non_owner:
            owner_type = 2
        if owner and non_owner:
            owner_type = 3

        record_to_ownership[record_id] = owner_type

    print("Finish calculating ownership")
    return record_to_ownership


def retrieve_record_to_committer_experience(records, repo_to_contributor_data, commit_to_username):
    print("Start retrieving committer 's experience...")
    record_to_average_experience = {}
    record_to_committer_experience = {}
    record_to_total_count = {}
    for record in records:
        repo = record.repo
        record_id = int(record.id)
        contributor_data = repo_to_contributor_data[repo]
        commit_timestamp = get_commit_timestamp(record)

        # default = 1 to avoid divide by zero
        record_to_average_experience[record_id] = 1
        if record_id not in commit_to_username:
            record_to_committer_experience[record_id] = 0
            continue

        committer = commit_to_username[record_id]
        total_count = 1

        developer_count = 0
        record_to_committer_experience[record_id] = 0

        for username, data in contributor_data.items():
            commit_count = get_commit_count_before(data, commit_timestamp)
            if username == committer:
                record_to_committer_experience[record_id] = commit_count
            if commit_count != 0:
                total_count += commit_count
                developer_count += 1

        average_experience = 1
        if developer_count != 0:
            average_experience = total_count/developer_count

        record_to_average_experience[record_id] = average_experience
        record_to_total_count[record_id] = total_count

    record_to_experience_file_path = os.path.join(directory, 'data/github_statistics/features/record_to_experience.csv')
    with open(record_to_experience_file_path, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['record_id', 'committer_experience', 'average_experience', 'total_count'])
        for record in records:
            record_id = int(record.id)
            csv_writer.writerow([record_id, record_to_committer_experience[record_id], record_to_average_experience[record_id],
                                 record_to_total_count[record_id]])

    print('Finish retrieving committer experience')


def do_calculate_experience():
    records = data_loader.load_records(os.path.join(directory, 'MSR2019/experiment/full_dataset_with_all_features.txt'))
    id_to_record = {}
    for record in records:
        record_id = int(record.id)
        id_to_record[record_id] = record

    commit_to_username = get_commit_to_username()

    repo_to_contributor_data, repo_to_total_count, repo_to_contributor_total_count \
        = get_repo_to_contributor_data(id_to_record)

    retrieve_record_to_committer_experience(records, repo_to_contributor_data, commit_to_username)


def do_calculate_ownership():
    records = data_loader.load_records(os.path.join(directory, 'MSR2019/experiment/full_dataset_with_all_features.txt'))
    id_to_record = {}
    for record in records:
        record_id = int(record.id)
        id_to_record[record_id] = record

    commit_to_username = get_commit_to_username()
    record_to_ownership = retrieve_record_to_ownership(records, commit_to_username)

    print("Writting to file...")
    record_to_ownership_file_path = os.path.join(directory, 'data/github_statistics/features/record_to_ownership.csv')
    with open(record_to_ownership_file_path, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['record_id', 'ownership_type'])
        for record in records:
            record_id = int(record.id)
            csv_writer.writerow(
                [record_id, record_to_ownership[record_id]])


def load_committer_experience():
    record_to_committer_experience, record_to_average_experience, record_to_total_count = {}, {}, {}
    experience_file_path = os.path.join(directory, 'data/github_statistics/features/record_to_experience.csv')
    with open(experience_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        count = 0
        for row in csv_reader:
            count += 1
            if count == 1:
                continue
            record_id = int(row[0])
            committer_exp = float(row[1])
            average_exp = float(row[2])
            total_count = int(row[3])
            record_to_committer_experience[record_id] = committer_exp
            record_to_average_experience[record_id] = average_exp
            record_to_total_count[record_id] = total_count

    return record_to_committer_experience, record_to_average_experience, record_to_total_count


def load_ownership():
    record_to_ownership = {}
    ownership_filepath = os.path.join(directory, 'data/github_statistics/features/record_to_ownership.csv')
    with open(ownership_filepath, 'r') as file:
        csv_reader = csv.reader(file)
        count = 0
        for row in csv_reader:
            count += 1
            if count == 1:
                continue
            record_id = int(row[0])
            ownership = int(row[1])
            record_to_ownership[record_id] = ownership

    return record_to_ownership


def get_first_commit_on_repo(data):
    min_time = max_timestamp
    for week_data in data['weeks']:
        if week_data['c'] <= 0:
            continue
        min_time = min(min_time, week_data['w'])

    return min_time


def get_test_file_count(record):
    count = 0
    commit = record.commit
    for commit_file in commit.files:
        file_name = commit_file.file_name
        if 'src/test' in file_name and file_name.endswith('.java'):
            count += 1

    return count


def get_main_file_count(record):
    count = 0
    commit = record.commit
    for commit_file in commit.files:
        file_name = commit_file.file_name
        if 'src/test' not in file_name and file_name.endswith('.java'):
            count += 1

    return count


def get_repo_to_repo_first_commit(repo_to_contributor_data):
    repo_to_repo_first_commit = {}

    for repo, contributor_data in repo_to_contributor_data.items():
        min_time = max_timestamp
        for username, data in contributor_data.items():
            time_stamp = get_first_commit_on_repo(data)
            min_time = min(min_time, time_stamp)
        repo_to_repo_first_commit[repo] = min_time

    return repo_to_repo_first_commit
