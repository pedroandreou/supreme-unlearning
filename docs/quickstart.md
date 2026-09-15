# Run your first experiment

Follow this example from the repository root after checking the
[platform prerequisites](environment_setup.md#prerequisites).

```bash
# 1. Clone
git clone https://github.com/pedroandreou/supreme-unlearning.git
cd supreme-unlearning

# 2. Set up environment - the Makefile is the entry point for local dev: it creates
#    the venv (named `unlearning` by default; override with VENV=<name>), installs the
#    pinned deps + SUPREME (editable), and enables the git hook. (Prompts if it
#    already exists; pass ON_EXISTING=reuse to skip.)
make cuda                  # NVIDIA GPU (Linux / WSL2).  Apple Silicon / CPU: `make mps`
source unlearning/bin/activate

# 3. Configure W&B + HF tokens
cp .env.example .env
# edit .env with your WANDB_KEY, WANDB_USERNAME and HUGGING_FACE_HUB_TOKEN

# 4. Small experiment - one seed, one dataset, retraining and two unlearning methods
bash src/supreme/run_local.sh \
  --gpu 0 --models ViT --training-seeds 260 \
  --methods retrain,finetune,ssd \
  --strategies random_ --datasets Cifar10 \
  --forget-percs 0.01
```

Full environment setup (Docker Dev Container, MPS prerequisites, etc.) is documented in [`docs/environment_setup.md`](environment_setup.md). The Docker image is NVIDIA-only (Linux / WSL2); macOS users follow the virtual-env path above.
