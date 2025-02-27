# **Docker TensorFlow Training Tutorial**

This tutorial provides a guide for setting up and running TensorFlow inside Docker.

---

## **1. Verify Docker Installation**
Check if Docker is installed:
```sh
docker --version
```
Check if Docker is running:
```sh
docker ps -a
```

---

## **2. List Docker Images**
To check available images:
```sh
docker images
```
**Output format:**
```
REPOSITORY          TAG       IMAGE ID        CREATED         SIZE
tensorflow/tensorflow  latest-gpu  abc123def456  3 days ago     5.6GB
```
- `{REPOSITORY}` – Image name (e.g., `tensorflow/tensorflow`)
- `{TAG}` – Version tag (e.g., `latest-gpu`)
- `{IMAGE ID}` – Unique ID of the image
- `{SIZE}` – Size of the image

**If TensorFlow is not available, pull it:**
```sh
docker pull tensorflow/tensorflow:latest-gpu
```

---

## **3. List Docker Containers**
Check running containers:
```sh
docker ps
```
Check **all** containers, including stopped ones:
```sh
docker ps -a
```

**Output format:**
```
CONTAINER ID   IMAGE      COMMAND        STATUS         NAMES
123abc456def   tensorflow/tensorflow "bash"        Up 10 minutes   tensorflow_container
```
- `{CONTAINER ID}` – Unique ID for the container
- `{IMAGE}` – Docker image used
- `{COMMAND}` – Command running inside the container
- `{STATUS}` – Running (`Up`), stopped (`Exited`), etc.
- `{NAMES}` – Auto-assigned or user-defined name

---

## **4. Run a New TensorFlow Docker Container**
### **With GPU Support & Port Forwarding**
```sh
docker run -it --gpus all -p 8888:8888 --name {NAME} tensorflow/tensorflow:latest-gpu bash
```
- `-it` – Interactive mode (keeps the terminal open)
- `--gpus all` – Enables GPU usage inside the container
- `-p 8888:8888` – Maps port `8888` from the container to the host (needed for Jupyter)
- `--name {NAME}` – Assigns a custom name (e.g., `tensorflow_container`)
- `tensorflow/tensorflow:latest-gpu` – Specifies the image

### **With Volume Mounting (For Persistent Data)**
To mount a local directory:
```sh
docker run -it --gpus all -p 8888:8888 -v /local/path:/workspace --name {NAME} tensorflow/tensorflow:latest-gpu bash
```
- `-v /local/path:/workspace` – Mounts the host directory `/local/path` into the container at `/workspace`

---

## **5. Start/Restart a Stopped Container**
If a container exists but is stopped, restart it:
```sh
docker start -ai {NAME}
```
- `-a` – Attach to the container (shows output)
- `-i` – Interactive mode

---

## **6. Install Dependencies Inside the Container**
Once inside the container, install required libraries:
```sh
pip install jupyterlab optuna autokeras
```

```sh
apt update && apt install -y git
```

Check if TensorFlow detects the GPU:
```sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## **7. Locate and Verify Notebooks**
List files inside the container:
```sh
ls /workspace
```
Move to the correct directory:
```sh
cd /workspace
```

---

## **8. Start Jupyter Notebook**
Run Jupyter with external access:
```sh
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
- `--ip=0.0.0.0` – Allows connections from any IP (required in Docker)
- `--port=8888` – Runs Jupyter on port `8888`
- `--allow-root` – Allows execution as `root` (required inside containers)

Copy the **access token** from the terminal output.

---

## **9. Access Jupyter Notebook**
On your **host machine**, open a browser and enter:
```
http://localhost:8888/?token={YOUR_TOKEN}
```
Replace `{YOUR_TOKEN}` with the token shown in the terminal.

---

## **10. Train Your Model**
Open `example.ipynb` and execute the cells.

---

## **11. Exit the Container**
After training, **exit** without stopping the container:
```sh
exit
```
To **stop** the container:
```sh
docker stop {NAME}
```

---

## **12. Save Container State**
If you want to keep modifications inside the container, **commit it as a new image**:
```sh
docker commit {CONTAINER ID} tensorflow_custom
```
Now, you can run it later with:
```sh
docker run -it --gpus all -p 8888:8888 --name {NEW_NAME} tensorflow_custom bash
```

---

## **13. Remove Unused Containers & Images**
List all stopped containers:
```sh
docker ps -a -f "status=exited"
```
Remove a specific container:
```sh
docker rm {CONTAINER ID}
```
Remove all stopped containers:
```sh
docker container prune -f
```
Remove an image:
```sh
docker rmi {IMAGE ID}
```
Remove all unused images:
```sh
docker image prune -a -f
```

---

## **14. Full Command Reference**
| Action                          | Command |
|---------------------------------|---------|
| Check Docker version | `docker --version` |
| List images | `docker images` |
| List all containers | `docker ps -a` |
| Run a new TensorFlow container | `docker run -it --gpus all -p 8888:8888 --name {NAME} tensorflow/tensorflow:latest-gpu bash` |
| Run container with volume | `docker run -it --gpus all -p 8888:8888 -v /local/path:/workspace --name {NAME} tensorflow/tensorflow:latest-gpu bash` |
| Start an existing container | `docker start -ai {NAME}` |
| Install Jupyter inside container | `pip install jupyterlab` |
| Start Jupyter Notebook | `jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root` |
| Stop a container | `docker stop {NAME}` |
| Exit a container | `exit` |
| Remove a container | `docker rm {CONTAINER ID}` |
| Remove all stopped containers | `docker container prune -f` |
| Remove an image | `docker rmi {IMAGE ID}` |
| Remove all unused images | `docker image prune -a -f` |
| Save container as image | `docker commit {CONTAINER ID} tensorflow_custom` |

---
If something breaks, use:
```sh
docker logs {CONTAINER ID}
```
To check what happened.