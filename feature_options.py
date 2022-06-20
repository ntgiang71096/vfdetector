class ExperimentOption():
    def __init__(self):
        self.data_set_size = -1
        self.ignore_number = True
        self.use_github_issue = True
        self.use_jira_ticket = True
        self.use_comments = False
        self.use_bag_of_word = True
        self.positive_weights = [0.5]
        self.max_n_gram = 1
        self.min_document_frequency = 1
        self.use_linked_commits_only = False

        # if self.use_issue_classifier = False, issue's information is attached to commit message
        self.use_issue_classifier = True

        self.fold_to_run = 10
        self.use_stacking_ensemble = True
        self.tf_idf_threshold = -1
        self.use_patch_context_lines = False
        self.unlabeled_size = -1