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
