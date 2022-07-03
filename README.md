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



### Prepare from Scratch
- To start using VulLinker, we need to clone this project
- Then, we can install the required libraries in the requirements.txt. For an easier virtual environment, we can use conda env.

```conda install --file requirements.txt```

Keep in mind when installing the requirements, it uses the NVidia Apex (https://github.com/NVIDIA/apex). If apex is not installed correctly when using requirements.txt, we need to manually install it. The command used for Windows:

```pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext```

While for linux:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- To prepare the dataset for the training and testing, we can use the script in “<working_directory>/Utilities>/data_preparation.py”. We can actually skip this step as the processed dataset is already included in the repository. To run the data preparation, we can use the following command:

```cd <working_directory>/Utilities>/
   python data_preparation.py
```

- Then, if we choose to run the dataset_preparation, we need to move the processed dataset into “<working_directory>/LightXML/data” folder.
- To run the training and evaluation script, we can use the run.sh script in the “<working_directory>/LightXML/” folder.

```./run.sh cve_data```


### Use Docker Image
For ease of use, we provide a docker image that can be accessed in:
https://hub.docker.com/repository/docker/ratnadira/vullinker
We can run and pull the docker image using below command:

```docker run --name=<docker_name> --gpus '"device=0,1"' --shm-size 32G -it -p 8000:8000 ratnadira/vullinker```

Noted that we can change the gpus parameter based on the spec that what we have.

### Datasets:

For TensorFlow dataset, please refer to: https://zenodo.org/record/6792205#.YsG03-xByw4

For SAP dataset, please refer to paper: "HERMES: Using Commit-Issue Linking to Detect Vulnerability-Fixing Commits"

### Training:

Message Classifier:

```python message_classifier.py --dataset_path <path_to_dataset> --model_path <saved_model_path> ```

Issue Classifier:

```python issue_classifier.py --dataset_path <path_to_dataset> --model_path <saved_model_path> ```

Patch Classifier:

- Finetuning: ```python vulfixminer_finetune.py --dataset_path <path_to_dataset> --finetune_model_path <saved_finetuned_model_path>```
- Training: ```python vulfixminer.py --dataset_path <path_to_dataset> --model_path <saved_model_file_path> --finetune_model_path <saved_finetuned_model_path> --train_prob_path <store_train_probability_to_path> --test_prob_path <store_test_probability_to_path>```

Ensemble Classifier:
```python variant_ensemble.py --config_file <path_to_config>```

Please follow our examples "tf_dataset.conf" or "sap_dataset.conf" for more details
