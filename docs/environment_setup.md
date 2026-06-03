# Environment Setup

Two supported flows: a **virtual environment** (recommended for stable multi-day training) and a **Docker Dev Container** (recommended for reproducible development).

## Prerequisites

- Python 3.9 (if `python3.9` isn't on your PATH, a version manager such as [pyenv](https://github.com/pyenv/pyenv) works - see §3a)
- One of: NVIDIA GPU with CUDA 12.1, Apple Silicon (MPS), or CPU
- Hugging Face Hub token (required for automatic ViT model downloads)
- Weights & Biases account + API key (evaluation results are logged exclusively to W&B)

## Platform support

| Install path | Linux + NVIDIA | Windows + NVIDIA (WSL2) | Apple Silicon Mac (M1/M2/M3/M4) | Intel Mac / other |
|---|---|---|---|---|
| **Virtual environment** (§3a) | ✅ `requirements/requirements.cuda_12_1.txt` | ✅ `requirements/requirements.cuda_12_1.txt` | ✅ `requirements/requirements.mps.txt` | CPU only (slow) |
| **Docker Dev Container** (§3b) | ✅ | ✅ via WSL2 + NVIDIA Container Toolkit | ❌ | ❌ |

The Docker image is built on a CUDA 12.1 base and the compose file requires `runtime: nvidia` - it will not start without an NVIDIA GPU visible to Docker. **Apple Silicon and any non-NVIDIA host must use the virtual environment path (§3a)**; Docker Desktop on macOS cannot expose the M-series GPU to containers.

## 1. Clone the repository

```bash
git clone https://github.com/pedroandreou/supreme-unlearning.git
```

## 2. Set up environment variables

Copy the template:

```bash
# Linux / macOS
cp .env.example .env

# Windows
xcopy .env.example .env
```

Update `.env` with your credentials.

**Required:**
- `HUGGING_FACE_HUB_TOKEN` - Hugging Face Hub token for ViT downloads
- `WANDB_KEY` and `WANDB_USERNAME` - results are logged exclusively to W&B; no standalone JSON/CSV export

**Optional (Docker Dev Container only):**
- `GITHUB_USERNAME`, `GITHUB_EMAIL` - used to configure git inside the container
- `SSH_KEY` - private key for `git push` over SSH from inside the container (auto-generated if left empty)

The Docker Dev container reads `.env` during the build, so create it **before** the build.

## 3a. Virtual environment (recommended)

The `Makefile` is the single entry point for setup: it **creates** the virtual
environment (named `unlearning` by default - hardware-neutral; override with
`VENV=<name>`), installs the pinned dependencies + SUPREME (editable), and
enables the pre-commit git hook. Pick the target matching your hardware:

```bash
make cuda      # NVIDIA GPU (CUDA 12.1) - Linux / WSL2
make mps       # Apple Silicon (MPS - M1/M2/M3/M4) / CPU
```

`make` builds the venv with a Python 3.9 interpreter, preferring `python3.9` on
your PATH and falling back to the newest installed pyenv 3.9.x. If neither is
available, or you want a specific interpreter, point it there explicitly:

```bash
make mps BASE_PYTHON=$HOME/.pyenv/versions/3.9.12/bin/python
```

If the venv directory already exists, `make` asks whether to **reuse** (reinstall
into it) or **delete and recreate** it. Skip the prompt with `ON_EXISTING=reuse`
or `ON_EXISTING=recreate` (e.g. in notebooks or CI).

A Makefile can't activate the venv for your shell, so activate it yourself for
interactive use (running experiments, `python`, etc.):

```bash
source unlearning/bin/activate      # or: source <your VENV name>/bin/activate
python --version                    # confirm 3.9.x
```

Run `make help` to list every target - e.g. `make quality` / `make style`
(lint + format), `make build` (sdist + wheel), `make clean`. The MPS requirements
file uses the standard PyPI PyTorch build, which includes MPS support natively;
`bitsandbytes` and `nvidia-ml-py` are omitted as they are CUDA-only.

SUPREME is also a regular pip package - published on PyPI as `supreme-unlearning`
and imported as `supreme` - so to reuse it from another project you can install
it directly (optionally with the CUDA extra for bitsandbytes precision and NVIDIA
telemetry, and/or the DeepSpeed extra for the ZeRO strategy):

```bash
pip install supreme-unlearning            # core (CPU / MPS)
pip install "supreme-unlearning[cuda]"    # + bitsandbytes, nvidia-ml-py (NVIDIA only; wheels, installs cleanly)
pip install "supreme-unlearning[deepspeed]"     # + deepspeed (compiles CUDA ops: needs a CUDA toolkit / CUDA_HOME, not just the driver)
pip install "supreme-unlearning[tensorboard]"   # + TensorBoard logger
```

> **DeepSpeed needs a CUDA toolkit, not just the driver.** It builds CUDA ops at
> install time and aborts with `CUDA_HOME does not exist` on hosts that only have
> the NVIDIA driver. Install a CUDA 12.1 toolkit and `export CUDA_HOME=...` first
> (e.g. `/usr/local/cuda-12.1`, or `$CONDA_PREFIX` after
> `conda install -c nvidia cuda-toolkit=12.1`). This is why DeepSpeed is excluded
> from the default `make cuda` install (`make deepspeed` opts in).

This installs the `supreme-train` and `supreme-unlearn` console scripts and the
importable public API (`import supreme`). When SUPREME is installed as a wheel
and run from outside the repo, set `SUPREME_PROJECT_ROOT` to a writable
directory so `logs/` and checkpoints are created there.

## 3b. Docker Dev Container (alternative)

> **Requires an NVIDIA GPU.** Linux or Windows + WSL2 only. Apple Silicon and other non-NVIDIA hosts: use §3a instead.

> **Host prerequisite: the NVIDIA Container Toolkit.** The compose services use
> `runtime: nvidia`, so the host needs more than the GPU driver - it needs the
> NVIDIA Container Toolkit, which registers the `nvidia` Docker runtime. Without
> it, `nvidia-smi` works on the host but containers fail with
> `unknown or invalid runtime name: nvidia` (compose) or
> `failed to discover GPU vendor from CDI` (`docker run --gpus all`). Install it
> once (Debian/Ubuntu; needs sudo):
>
> ```bash
> curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
>   | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
> curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
>   | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
>   | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
> sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
> sudo nvidia-ctk runtime configure --runtime=docker   # registers the nvidia runtime
> sudo systemctl restart docker
> docker run --rm --gpus all ubuntu nvidia-smi          # sanity check: should list your GPUs
> ```

> **On a shared / managed GPU host (no admin rights)? Use §3a instead.** The
> NVIDIA Container Toolkit is a **host-daemon** component: it must be installed on
> the host that runs `dockerd`, and it cannot be installed from inside a container
> (the dev container's passwordless sudo only grants root *within* the container,
> not on the host). On a cluster where your account can't install host packages -
> e.g. `sudo -l` shows only a couple of whitelisted commands - you cannot enable
> GPU-in-Docker yourself; ask the cluster admins to run the block above, **or skip
> Docker entirely and use the virtual environment (§3a)**. The venv talks to the
> NVIDIA driver directly and needs no container toolkit, so it runs on the GPUs
> with no elevated privileges. Confirm with the GPU self-check below.

With VS Code and Docker installed and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), open the Command Palette (`View → Command Palette`) and run **Dev Containers: Reopen in Container**. You should see a prompt like this:

![Dev Container Initial Prompt](../assets/dev-container-initial-prompt.png)

By default this **pulls a prebuilt image** from the GitHub Container Registry (`ghcr.io/pedroandreou/supreme-unlearning-devcontainer`, published by [`.github/workflows/devcontainer.yml`](../.github/workflows/devcontainer.yml)), so the container is ready as fast as it downloads, with no local build. The `postCreateCommand` still runs on first open to finish provisioning.

**Prefer to build locally instead?** In [`docker/docker-compose.dev.yml`](../docker/docker-compose.dev.yml) change the extended service from `cuda_12_1_devcontainer` back to `cuda_12_1_from_scratch`, then reopen. The first build takes several minutes; subsequent builds use Docker cache.

If the prompt has disappeared, use `View → Command Palette → Developer: Reload Window`, or `Dev Containers: Rebuild Without Cache and Reopen in Container`. You may need to rebuild if CUDA becomes unavailable after extended use.

**Why we recommend the virtual environment for long runs:** the NVIDIA container runtime occasionally fails with *"Failed to initialize NVML: Unknown Error"* after prolonged GPU access. The official workaround is to set `no-cgroups = false` in `/etc/nvidia-container-runtime/config.toml`, which requires sudo on the host. The virtual environment avoids this entirely.

## 4. macOS-only: install bash 4+

`src/supreme/run_local.sh` uses bash 4 features (`mapfile`, etc.). macOS ships with bash 3.2.

```bash
brew install bash
# Then run experiments with:
/opt/homebrew/bin/bash src/supreme/run_local.sh --gpu 0 ...
```

## Verifying the install

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
python -c "import supreme; print('SUPREME', supreme.__version__, '| API:', 'register_unlearning_method' in dir(supreme))"
python -c "from supreme.utils.unlearning import unlearn_main; print('SUPREME import OK')"
```

**GPU self-check (NVIDIA hosts).** Confirms PyTorch sees and can use the GPUs -
works the same in the venv (§3a) or inside the container (§3b):

```bash
python -c "import torch; n=torch.cuda.device_count(); print('cuda', torch.cuda.is_available(), '| cuda ver', torch.version.cuda, '| GPUs', n, '|', [torch.cuda.get_device_name(i) for i in range(n)])"
```

`cuda True` with a non-empty GPU list means you're ready to train. `cuda False`
on a machine whose `nvidia-smi` shows GPUs usually means a CPU-only PyTorch build
(reinstall via `make cuda`) or, inside Docker, that the container was started
without GPU access (see the NVIDIA Container Toolkit note in §3b).

Then run a minimal smoke test - see [README → Running Experiments](../README.md#-running-experiments).
