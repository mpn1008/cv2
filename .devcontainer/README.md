Usage
-----

1. Ensure Docker is installed and the host has NVIDIA drivers.
2. Install the NVIDIA container toolkit on the host (Ubuntu example):

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3. In VS Code, open the Command Palette and choose "Dev Containers: Reopen in Container".
4. The container is built from the included `Dockerfile` and started with GPU passthrough (`--gpus all`).

Notes
-----
- The container image is based on `nvidia/cuda:12.8.2-devel-ubuntu22.04`.
- `postCreateCommand` upgrades `pip` inside the container.
- If you already have a compatible `image` you'd like to use, update `devcontainer.json` accordingly.
