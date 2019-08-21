# Deep Image Prior

Single Image Super Resolution using Deep Image Prior

## Overview


## Paperspace

To train models, the service used was [Paperspace](https://www.paperspace.com). The script `submit_job.sh` is used to submit a job to paperspace. It requires three environment variables to be set, `CONTAINER_NAME, DOCKERHUB_USERNAME, DOCKERHUB_PASSWORD`. The dockerhub username and password are necessary because the repository used is private. If it is public, edit the script accordingly.

In order to use it, a docker container must be created and hosted on a platform such as [DockerHub](https://hub.docker.com). The script, `Dockerfile` can be used to build this docker container. If you are using DockerHub, run the commands

```bash
docker build -t <hub-user>/<repo-name>[:tag] .
docker push <hub-user>/<repo-name>[:tag]
```

This will create a docker container and push it to the required repository. This docker container is then utilized by the script used to submit a training job