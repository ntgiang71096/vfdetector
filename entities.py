import json
from json import JSONEncoder


class Record():
    def __init__(self, id=None, repo=None, commit_id=None, commit_message=None, label=None, json_value=None):
        # assert ((json is not None)
        #         or (id is not None and commit_id is not None
        #             and commit_message is not None)), "invalid Record construction"
        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
            if self.commit is not None:
                self.commit = GithubCommit(json_value=json.dumps(self.commit))

            github_issue_list = []
            if self.github_issue_list is not None and len(self.github_issue_list) > 0:
                for github_issue_dict in self.github_issue_list:
                    github_issue_list.append(GithubIssue(json_value=json.dumps(github_issue_dict)))
                self.github_issue_list = github_issue_list

            jira_ticket_list = []
            if self.jira_ticket_list is not None and len(self.jira_ticket_list) > 0:
                for jira_ticket_dict in self.jira_ticket_list:
                    jira_ticket_list.append(JiraTicket(json_value=json.dumps(jira_ticket_dict)))
                self.jira_ticket_list = jira_ticket_list

        else:
            self.id = id
            self.repo = repo
            self.commit_id = commit_id
            self.commit_message = commit_message
            self.label = label
            self.jira_ticket_list = []
            self.github_issue_list = []
            self.commit = None
            self.branch = None
        self.issue_info = None
        self.code_terms = []
        self.text_terms_parts = []

    def __repr__(self):
        return "{}/commit/{}".format(self.repo, self.commit_id)

    def add_jira_ticket(self, jira_ticket):
        self.jira_ticket_list.append(jira_ticket)

    def add_github_ticket(self, github_issue):
        self.github_issue_list.append(github_issue)

    def set_commit(self, commit):
        self.commit = commit


class JiraTicket():
    def __init__(self, json_value=None, name=None, summary=None,
                 description=None, created_at=None, creator=None, assignee=None,
                 fix_versions=None, issue_type=None, priority=None,
                 resolution=None, resolution_date=None, status=None, comments=None):

        assert ((json_value is not None) or (name is not None and
                                             summary is not None and created_at is not None
                                             and creator is not None)), "Invalid construction for JiraTicket"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
            if self.fix_versions is not None:
                self.fix_versions = JiraTicketFixVersion(json_value=json.dumps(self.fix_versions))

            if self.issue_type is not None:
                self.issue_type = JiraTicketIssueType(json_value=json.dumps(self.issue_type))

            if self.priority is not None:
                self.priority = JiraTicketPriority(json_value=json.dumps(self.priority))

            if self.resolution is not None:
                self.resolution = JiraTicketResolution(json_value=json.dumps(self.resolution))

            if self.status is not None:
                self.status = JiraTicketStatus(json_value=json.dumps(self.status))

            comment_dict_list = self.comments
            comments = []
            for comment_dict in comment_dict_list:
                comment = JiraTicketComment(json_value=json.dumps(comment_dict))
                comments.append(comment)
            self.comments = comments

        else:
            # summary is title of jira ticket
            self.name = name
            self.summary = summary
            self.description = description
            self.created_at = created_at
            self.creator = creator
            self.assignee = assignee
            self.fix_versions = fix_versions
            self.issue_type = issue_type
            self.priority = priority
            self.resolution = resolution
            self.resolution_date = resolution_date
            self.status = status

            self.comments = comments

            self.code_terms = []
            self.text_terms_parts = []
            self.id = 0

    def __repr__(self):
        return "Jira ticket: {} \n\n" \
               "description: {} \n\n" \
               " created at: {}".format(self.summary, self.description, self.created_at)


class JiraTicketPriority:
    def __init__(self, json_value=None, priority_id=None, priority_name=None):
        assert ((json_value is not None) or (priority_id is not None and priority_name is not None)), \
            "Invalid construction for JiraTicketPriority"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.priority_id = priority_id
            self.priority_name = priority_name


class JiraTicketFixVersion:
    def __init__(self, json_value=None, name=None, release_date=None):
        assert (json_value is not None or name is not None), \
            "Invalid construction for JiraTicketFixVersion"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.name = name
            self.release_date = release_date


class JiraTicketIssueType:
    def __init__(self, json_value=None, name=None, description=None):
        assert ((json_value is not None) or (name is not None and description is not None)), \
            "Invalid consturction for JiraTicketIssueType"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.name = name
            self.description = description


class JiraTicketStatus:
    def __init__(self, json_value=None, name=None, description=None, category=None):
        assert ((json_value is not None) or (name is not None and description is not None and category is not None)), \
            "Invalid construction for JiraTicketStatus"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.name = name
            self.description = description
            self.category = category


class JiraTicketResolution:
    def __init__(self, json_value=None, resolution_id=None, name=None, description=None):
        assert ((json_value is not None)
                or (resolution_id is not None and name is not None and description is not None)), \
            "Invalid construction for JiraTicketResolution"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.resolution_id = resolution_id
            self.name = name
            self.description = description


class JiraTicketComment:
    def __init__(self, json_value=None, created_by=None, body=None, created_at=None, updated_at=None):
        assert ((json_value is not None) or (body is not None and created_at is not None and updated_at is not None)), \
            "Invalid construction for JiraTicketComment"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.created_by = created_by
            self.body = body
            self.created_at = created_at
            self.update_at = updated_at


class GithubIssue():
    def __init__(self, json_value=None, title=None, body=None,
                 author_name=None, created_at=None, closed_at=None, closed_by=None,
                 last_modified=None, comments=None):
        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
            comment_dict_list = self.comments
            comments = []
            for comment_dict in comment_dict_list:
                comments.append(GithubIssueComment(json_value=json.dumps(comment_dict)))
            self.comments = comments
        else:
            self.title = title
            self.body = body
            self.author_name = author_name
            self.created_at = created_at
            self.closed_at = closed_at
            self.closed_by = closed_by
            self.last_modified = last_modified
            self.comments = comments

    def __repr__(self):
        return "Github issue: " + self.title + ", created at " + self.created_at


class GithubIssueComment():
    def __init__(self, json_value=None, body=None, created_at=None, created_by=None, last_modified=None):
        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.body = body
            self.created_at = created_at
            self.created_by = created_by
            self.last_modified = last_modified

    def __repr__(self):
        return "Github issue comment at " + self.created_at + " :" + self.body


class GithubCommit:
    def __init__(self, json_value=None, author_name=None, created_date=None, files=None):
        # assert ((json_value is not None)
        #         or (author_name is not None and
        #             created_date is not None and files is not None)), "Invalid construction for GitHubCommit"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
            commit_files = []
            for file in self.files:
                json_value = json.dumps(file)
                commit_file = GithubCommitFile(json_value=json_value)
                commit_files.append(commit_file)
            self.files = commit_files
        else:
            self.author_name = author_name
            self.created_date = created_date
            self.files = files

    def __repr__(self):
        return "Author: " + self.author_name + ", created date: " + self.created_date


# if file is binary, there will be no patch
class GithubCommitFile:
    def __init__(self, json_value=None, file_name=None, patch=None, status=None, additions=None, deletions=None,
                 changes=None):
        # assert ((json_value is not None)
        #         or (file_name is not None and status is not None
        #             and deletions is not None and changes is not None)), "Invalid construction for GithubCommitFile"

        if json_value is not None:
            self.__dict__.update(json.loads(json_value))
        else:
            self.file_name = file_name
            self.patch = patch
            self.status = status
            self.additions = additions
            self.deletions = deletions
            self.changes = changes

    def __repr__(self):
        return "file name: " + self.file_name


class EntityEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Record) \
                or isinstance(o, GithubIssue) \
                or isinstance(o, JiraTicket) \
                or isinstance(o, JiraTicketResolution) \
                or isinstance(o, JiraTicketIssueType) \
                or isinstance(o, JiraTicketPriority) \
                or isinstance(o, JiraTicketFixVersion) \
                or isinstance(o, JiraTicketStatus) \
                or isinstance(o, JiraTicketComment) \
                or isinstance(o, GithubIssueComment) \
                or isinstance(o, GithubCommit) \
                or isinstance(o, GithubCommitFile):
            return o.__dict__
        else:
            return JSONEncoder.encode(self, o)
