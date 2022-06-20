import xml.dom.minidom
import pandas as pd


def get_title(vuln):
    return vuln.getElementsByTagName('Title')[0].childNodes[0].data


def get_description(vuln):
    note = vuln.getElementsByTagName('Notes')[0].getElementsByTagName('Note')[0]
    type = note.attributes['Type'].value
    if type == 'Description':
        return note.childNodes[0].data
    else:
        return ''


def get_commit_list(vuln):
    references = vuln.getElementsByTagName('References')[0].getElementsByTagName('Reference')

    return [ref.getElementsByTagName('URL')[0].childNodes[0].data for ref in references]


def is_tensor_fix(url):
    return url.startswith('https://github.com/tensorflow/tensorflow/commit/')


def collect_all_tensor_fixes():

    cve_file_list = [
        'cve/allitems-cvrf-year-2022.xml',
        'cve/allitems-cvrf-year-2021.xml',
        'cve/allitems-cvrf-year-2020.xml',
        'cve/allitems-cvrf-year-2019.xml',
        'cve/allitems-cvrf-year-2018.xml',
        'cve/allitems-cvrf-year-2017.xml',
        'cve/allitems-cvrf-year-2016.xml',
        'cve/allitems-cvrf-year-2015.xml',
        'cve/allitems-cvrf-year-2014.xml',
        'cve/allitems-cvrf-year-2013.xml',
        'cve/allitems-cvrf-year-2012.xml',
        'cve/allitems-cvrf-year-2011.xml',
        'cve/allitems-cvrf-year-2010.xml'
    ]

    tensor_fixes = []
    url_set = set()
    count = 0
    for cve_file in cve_file_list:
        doc = xml.dom.minidom.parse(cve_file)
        vuln_list = doc.getElementsByTagName('Vulnerability')

        for vuln in vuln_list:
            title = get_title(vuln)
            description = get_description(vuln)
            commit_list = get_commit_list(vuln)
            for commit in commit_list:
                if is_tensor_fix(commit):
                    if commit not in url_set:
                        url_set.add(commit)
                        print(commit)
                        count += 1
                        print(count)
                        tensor_fixes.append((title, description, commit))

    tensor_fixes_file = 'tf_fixes.csv'

    df = pd.DataFrame(tensor_fixes, columns=['cve', 'description', 'commit_url'])

    df.to_csv(tensor_fixes_file, index=False)


if __name__ == '__main__':
    collect_all_tensor_fixes()
