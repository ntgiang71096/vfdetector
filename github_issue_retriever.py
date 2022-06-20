import github
import json

tokens = ["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",
          "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",
          "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",
          "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",
          "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",
          "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",
          "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"]


def retrieve_github_issues(state):
    token_index = 1
    current_page = 1277
    done = False
    while not done:
        file_path = 'github_issues/state_' + state + '_page_' + str(current_page) + '.json'
        gh = github.Github(tokens[token_index])
        repo = gh.get_repo('tensorflow/tensorflow')
        issues = repo.get_issues(state=state).get_page(current_page)
        data = []
        if len(issues) == 0:
            done = True
        for issue in issues:
            print(issue.number)
            comments = []
            for comment in issue.get_comments():
                comments.append(comment.body)

            item = {'number': issue.number, 'title': issue.title, 'body': issue.body, 'comments': comments}
            data.append(item)
            # break

        with open(file_path, 'w') as file:
            json.dump(data, file)
        current_page += 1
        token_index += 1
        if token_index == 7:
            token_index = 0


retrieve_github_issues('closed')
print()