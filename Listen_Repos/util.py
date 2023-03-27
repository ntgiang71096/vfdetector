import sqlite3
from datetime import datetime, timedelta
import pytz
import csv


def convert_time_to_UST(commit_time, time_zone):
    # Convert commit time string to datetime object
    dt = datetime.strptime(str(commit_time)[:-6], "%Y-%m-%d %H:%M:%S")

    # Create a timezone object based on the offset
    tz = pytz.FixedOffset(int(time_zone)/60)

    # Convert the datetime object to UTC
    commit_time_utc = dt.astimezone(pytz.utc)

    # Print the UTC time
    # print(commit_time_utc)

    return str(commit_time_utc)[:-6]

def database_execution():

    conn=sqlite3.connect('vulnerability_tensorflow.db')
    cur = conn.cursor()

    return cur

# init the dataset, including the commit time, its hash value, message info, and the prediction produced by our tool indicating whether the commit is for vulnerability-fixing or not
def initiate_database():

    cur = database_execution()
    
    try:
        cur.execute('DROP TABLE commits;')
    except:
        print('no table named commits')

    cur.execute('CREATE TABLE commits (committ_time TEXT, id TEXT, commit_message TEXT, label TEXT)')

# write the data into the database
def write2database(commit_time, id, msg, prediction):
    conn=sqlite3.connect('vulnerability_tensorflow.db')
    cur = conn.cursor()
    cur.execute("INSERT INTO commits (committ_time, id, commit_message, label) VALUES (?, ?, ?, ?)", (commit_time, id, msg, prediction))
    conn.commit()
    conn.close()

def write2csv(commit_time, id, msg, prediction):
    with open('log.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow([commit_time, id, msg, prediction])
    

# read the database
def checkdatabase():
    conn=sqlite3.connect('vulnerability_tensorflow.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM commits")
    results = cur.fetchall()
    print(results)