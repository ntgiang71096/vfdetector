from pydriller import Repository
import subprocess
import json
from util import write2database, checkdatabase, convert_time_to_UST, write2csv
from datetime import datetime, timedelta
import os

def listen_body(url):

    # Define the URL of the GitHub repository you want to retrieve commits from

    # Define the starting date for the analysis (e.g., the last 7 days)
    since_date = datetime.today() - timedelta(seconds=900)

    # Retrieve the most recent commits of the repository using PyDriller
    commits = Repository(url, since=since_date).traverse_commits()
    # print(len(commits))
    commit_patch = []
    num_commit = 0

    for commit in commits:
        commit_json = {}
        num_commit += 1
        commit_time, hash, message = commit.committer_date , commit.hash, commit.msg
        comit_time_zone = commit.committer_timezone

        # print diff of the commit
        # print(commit.diff)
        for file in commit.modified_files:
            diff_lines = file.diff.splitlines()
            commit_patch.extend(diff_lines)
        commit_json['id'] = hash
        commit_json['message'] = message        
        commit_json['patch'] = commit_patch
        # commit_json = json.dumps(commit_json)

        if commit_patch:

            working_dir = '../'

            with open("data.json", "w") as outfile:
                json.dump([commit_json], outfile)

            output = subprocess.check_output(['python', 'application.py', '-mode','prediction','-input','Listen_Repos/data.json','-threshold','0.5','-output','Listen_Repos/output.json'], cwd=working_dir)

            # print(commit_time)
            # print(comit_time_zone)
            # print(convert_time_to_UST(commit_time,comit_time_zone))
            # print('======\n\n\n')

            # read json
            with open('output.json', 'r') as file:
                data = file.read()
            json_data = json.loads(data)

            write2database(convert_time_to_UST(commit_time,comit_time_zone), commit_json['id'], commit_json['message'], json_data[0]['prediction'])
            write2csv(convert_time_to_UST(commit_time,comit_time_zone), commit_json['id'], commit_json['message'], json_data[0]['prediction'])


            # checkdatabase()
