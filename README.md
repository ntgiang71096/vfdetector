# VulCurator: A Vulnerability-fixing Commit Detector

## Tool Description

Here is a tool for automatically listening to github repos and analyze each commit in real-time to identify whether it is relevant to vulnerability-fixing.

To run this tool, you need to first set the appropriate environment variables as described in the second section of the tool's documentation. Then, you can provide the GitHub repository link and the desired listening period as inputs to the below Python script:

```
Listen_Repos/init.py -link [Github URL] -listenperiod [listening period in terms of seconds]
```

and then simply run this script. The newly generated commits as well as their classification results are stored in the local databases and txt files. 


The tool uses the GitHub API to fetch information about the newly published commits (i.e., commit hash, commit message, code changes, and time of commit). It then applies VulCurator algorithm to determine whether each commit is likely to contain vulnerability fixes. Each data as well as the classification results are stored in the local databases and txt files. If a commit is identified as relevant, the tool allows triggering further actions, such as notifying a security team or creating an issue in a vulnerability tracking system (still working on it).


# Setting VulCurator

Users can choose whether to prepare VulCurator from scratch or use the docker image. For ease of use, users can choose to use docker image as the required libraries are already installed to ensure that VulCurator can run smoothly.

### Use Docker Image

For ease of use, we provide a docker image that can be accessed in:
https://hub.docker.com/r/nguyentruongggiang/vfdetector

User can pull the docker image using below command:

``docker pull nguyentruongggiang/vfdetector:v1``

Run docker image:

``docker run --name vfdetector -it --shm-size 16G --gpus all nguyentruongggiang/vfdetector:v1``

Next, Move to VulCurator's working directory:

``cd ../VFDetector``

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

### Datasets:

For TensorFlow dataset, please refer to: https://zenodo.org/record/6792205#.YsG03-xByw4

For SAP dataset, please refer to paper: "HERMES: Using Commit-Issue Linking to Detect Vulnerability-Fixing Commits"

### Training:

Message Classifier:

``python message_classifier.py --dataset_path <path_to_dataset> --model_path <saved_model_path> ``

Issue Classifier:

``python issue_classifier.py --dataset_path <path_to_dataset> --model_path <saved_model_path> ``

Patch Classifier:

- Finetuning: ``python vulfixminer_finetune.py --dataset_path <path_to_dataset> --finetune_model_path <saved_finetuned_model_path>``
- Training: ``python vulfixminer.py --dataset_path <path_to_dataset> --model_path <saved_model_file_path> --finetune_model_path <saved_finetuned_model_path> --train_prob_path <store_train_probability_to_path> --test_prob_path <store_test_probability_to_path>``

Ensemble Classifier:
``python variant_ensemble.py --config_file <path_to_config>``

Please follow our examples "tf_dataset.conf" or "sap_dataset.conf" for more details

### Replicate our result:

Before, please download SAP dataset: https://drive.google.com/file/d/1NyCnXGD4VyVDZ2TMhqv4bDqYl14HDRUD/view?usp=sharing
and put it in working directory

Next, please follow our instructions to replicate our experimental results:

For Tensorflow dataset:

To train message classifier:
``python message_classifier.py --dataset_path tf_vuln_dataset.csv --model_path model/tf_message_classifier.sav``

To train issue classifier
``python issue_classifier.py --dataset_path tf_vuln_dataset.csv --model_path model/tf_issue_classifier.sav``

To finetune CodeBERT for patch classifier:
``python vulfixminer_finetune.py --dataset_path tf_vuln_dataset.csv --finetune_model_path model/tf_patch_vulfixminer_finetuned_model.sav``

To traing patch classifier:
``python vulfixminer.py --dataset_path tf_vuln_dataset.csv --model_path model/tf_patch_vulfixminer.sav --finetune_model_path model/tf_patch_vulfixminer_finetuned_model.sav --train_prob_path probs/tf_patch_vulfixminer_train_prob.txt --test_prob_path probs/tf_patch_vulfixminer_test_prob.txt``

To run ensemble classifier:
``python variant_ensemble.py --config_file tf_dataset.conf``

Similarly, for SAP dataset:

To train message classifier:
``python message_classifier.py --dataset_path sub_enhanced_dataset_th_100.txt --model_path model/sap_message_classifier.sav``

To train issue classifier
``python issue_classifier.py --dataset_path sub_enhanced_dataset_th_100.txt --model_path model/sap_issue_classifier.sav``

To finetune CodeBERT for patch classifier:
``python vulfixminer_finetune.py --dataset_path sap_patch_dataset.csv --finetune_model_path model/sap_patch_vulfixminer_finetuned_model.sav```

To traing patch classifier:
``python vulfixminer.py --dataset_path sap_patch_dataset.csv --model_path model/sap_patch_vulfixminer.sav --finetune_model_path model/sap_patch_vulfixminer_finetuned_model.sav --train_prob_path probs/sap_patch_vulfixminer_train_prob.txt --test_prob_path probs/sap_patch_vulfixminer_test_prob.txt``

To run ensemble classifier:
``python variant_ensemble.py --config_file sap_dataset.conf``
