# VulCurator: A Vulnerability-fixing Commit Detector


## Preparation
Users can choose whether to prepare VulCurator from scratch or use the docker image. For ease of use, users can choose to use docker image as the required libraries are already installed to ensure that VulCurator can run smoothly.

### Use Docker Image
For ease of use, we provide a docker image that can be accessed in:
https://hub.docker.com/r/nguyentruongggiang/vfdetector

User can pull the docker image using below command:

```docker pull nguyentruongggiang/vfdetector:v1```

Run docker image:

```docker run --name vfdetector -it --shm-size 16G --gpus all nguyentruongggiang/vfdetector:v1```

Next, Move to VulCurator's working directory:

```cd ../VFDetector```

Noted that we can change the gpus parameter based on the spec that what we have.

## Run VulCurator

### Prepare Input

- In order to run VulCurator, user must provide commits' info followed our predefined Json format:

```json
[
    {
        {
        "id": <commit_id>, 
        "message": <commit_message>,
        "issue": {
            "title": <issue_title>,
            "body": <issue_body>,
            "comments" : [<list_of_comments]
        },
        "patch": [list_of_code_change]
    },
  ...
]
```

The issue's information is optional.

Output of VulCurator is a json file with format depending on the selected mode.

### Prediction Mode

In Prediction Mode, given the input of a dataset of commits, VulCurator returns a list of likely vulnerability fixing commits along with the confidence scores. 
Note that, although VulCurator sets the classification threshold at 0.5 by default, VulCurator still allows users to adjust the threshold.

```
python application.py -mode prediction -input <path_to_input> -threshold <threshold> -output <path_to_output>
```

### Ranking Mode 
In the ranking mode, users can input data following our format and then VulCurator will output the sorted list of commits based on the probability that the commits are for vulnerability-fixing. Particular, users can use the following commands:

```
python application.py -mode ranking -input <path_to_input> -output <path_to_output>
```
