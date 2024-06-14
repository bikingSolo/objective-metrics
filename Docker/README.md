# Docker support

## Running

All Dockerfiles represent an <code>objective_metrics_run</code> script, that will be run in a docker **isolated** enviroment.

For it's proper work, you have to mount a folder with <code>datasets_config</code> folder (see docs) inside a container. Docker process will take configs out of it and will also write logs inside the mounted folder. Inside a docker process, this folder will be called <code>/app</code>.

Also, you will need to mount datasets to the container. All paths to datasets are written inside of configs, so you can either initially write this paths relatively to docker container's file system or mount datasets in such a way that all paths would match in both places. This is up to you.

<u>Example workspace for further examples:</u>

```
workspace/
├── dataset_configs/
    ├─ ...
    ├─ live-wcvqd_list.json  
    ├─ live-wcvqd_pairs.json  
    ├─ live-wcvqd.yaml  
    ├─ ...
    ├─ tid2013_list.json  
    ├─ tid2013_pairs.json  
    ├─ tid2013.yaml
    ├─ ...  
    ├─ live-vqc_list.json
    ├─ live-vqc.yaml  
    ├─ ...
├── ...
```


## Dockerfile_host-cuda_cpu-decord

With this Dockerfile you can build an image, that will contain objective-metrics package and all its dependencies.

Decord package will be installed via pip **without** any GPU support.

nvidia/cuda image is used as a base image, so cuda and nvidia drivers from host machine will be mounted inside a running container. You will need to have preinstalled **cuda** and **nvidia-container-toolkit** on the host machine to use it.

You **can** run this image on a host without CUDA and GPU, it will just work on CPU.

Size of a resulting image is about 17 Gb.

<u>Example usage:</u>

```
docker run \
--gpus all \ 
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v ./workspace:/app \
-v <pth_to_datasets>:<pth_to_datasets> \
<image_name> tid2013 live-vqc ...
```

## Dockerfile_host-cuda_gpu-decord

Everythin is the same with [Dockerfile_host-cuda_cpu-decord](#dockerfile_host-cuda_cpu-decord), except for decord package buildng.

Now decord is been building from source code to have a **NVDEC** support.

You **can't** run this image on a host without CUDA and GPU.

To successfully build this image, you will have to place a <code>libnvcuvid.so</code> file inside a dir with Dockerfile. It is beacuse nvidia drives can't be accesed during building part, only when running, due to limitations of nvidia/cuda base image.

Size of a resulting image is about 17 Gb.

<u>Example usage:</u>

```
docker run \
--gpus all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e DECORD_GPU=1 \
-v ./workspace:/app \
-v <pth_to_datasets>:<pth_to_datasets> \
<image_name> tid2013 live-vqc ...
```

## Dockerfile_within-cuda_gpu-decord

With this Dockerfile you can build an image, that will contain objective-metrics package and all its dependencies.

Decord is been building from source.

Here, CUDA is been installed **inside the image**, so host doesn't need to have CUDA and nvidia-container-toolkit. Only GPU is mandatory. You will have to provide your nvidia devices to a contaier with --device option. 

To inspect your nvidia devices, run: `$ ls /dev/*nvidia*`

This image will have the same nvidia-driver version as a host machine, so it is possible that you will need to replace 12.0 CUDA toolkit with another version, that mathces your nvidia-driver.

Now, you **don't need** to to place a <code>libnvcuvid.so</code> file inside a dir with Dockerfile, because all CUDA drivers are already inside of an image.

Image, builded with this Dockerfile will be **significantly** larger, than previous (about 25 Gb).

<u>Example usage:</u>

```
docker run \
--device /dev/nvidia0 \
--device /dev/nvidiactl \
--device /dev/nvidia-modeset \
--device /dev/nvidia-uvm \
--device /dev/nvidia-uvm-tools \
-e DECORD_GPU=1 \
-v ./workspace:/app \
-v <pth_to_datasets>:<pth_to_datasets> \
<image_name> tid2013 live-vqc ...
```