# Running on Windows: demo first, real experiments second

`/workspace/reproducer` belongs to the cloud machine where the code was written. It does
not refer to a directory on your Windows computer. Git Bash and the UCRT64/MSYS2 terminal
do not automatically connect to that machine.

The code is on branch **`codex/clm-history-dependence`**. Select that branch on GitHub or
clone it explicitly. A plain clone of the default `main` branch does not select this branch.

## 1. Run the offline demo in Git Bash or UCRT64

### Install Windows Python

Install Python 3.12 or newer from [python.org](https://www.python.org/downloads/windows/),
including the Python launcher. Reopen your terminal, then check:

```bash
git --version
py -3 --version
```

The Python version must be at least 3.11. These instructions use the Windows `py` launcher
to avoid accidentally choosing the MSYS2/UCRT Python interpreter.

### Download the branch

For a new checkout:

```bash
cd ~
git clone --branch codex/clm-history-dependence https://github.com/isClaptonGod/reproducer.git
cd reproducer
ls
```

You should see `pyproject.toml`, `clm_experiment`, `configs`, and `README.md`.

If you already cloned the repository into `~/reproducer`, use this instead:

```bash
cd ~/reproducer
git fetch origin
git switch codex/clm-history-dependence
```

If Git reports local changes that prevent switching, preserve those changes before switching;
do not reset or discard them to follow this guide.

### Install and run

Run these commands in the repository folder:

```bash
py -3 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e .
./.venv/Scripts/python.exe -m clm_experiment demo --out runs/demo
```

You do not need to activate the environment. Use `Scripts/python.exe` on native Windows;
`bin/python` is the Linux/WSL location.

Successful output ends with:

```text
Offline synthetic fixture finished: runs/demo/report.json
```

Open the report:

```bash
notepad.exe runs/demo/report.json
```

The report should contain `"synthetic": true` and `"run_count": 6`. Inconclusive decisions
are expected. This demo uses a scripted fixture, not Qwen or Claude. It needs neither
Docker nor API keys. Installing the package initially needs internet access.

To run it again, choose a new output folder:

```bash
./.venv/Scripts/python.exe -m clm_experiment demo --out runs/demo-2
```

## 2. Set up WSL2 for real experiments

The real Docker runner was tested on Linux. Use WSL2 Ubuntu rather than the UCRT64/MSYS2
terminal for these steps. The native Windows demo above does not establish native Windows
Docker support.

1. Open **PowerShell as Administrator** and install Ubuntu:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

2. Restart if Windows requests it. Launch **Ubuntu** from the Start menu and finish its
   username/password setup.
3. Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/).
   Enable its WSL2 backend and enable the Ubuntu distribution under
   **Settings → Resources → WSL Integration**. Keep Docker Desktop running.
4. In the **Ubuntu terminal**, run:

   ```bash
   sudo apt update
   sudo apt install -y git python3 python3-venv
   docker version
   ```

`docker version` must show both a client and a server. If the server cannot be reached,
finish Docker Desktop/WSL integration before continuing.

### Make a separate Linux checkout and environment

In Ubuntu:

```bash
cd ~
git clone --branch codex/clm-history-dependence https://github.com/isClaptonGod/reproducer.git
cd reproducer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
docker pull python:3.12-slim
docker build -t clm-work:0.1 .
```

Windows and WSL virtual environments are different; do not reuse the Windows `.venv`.
The ordinary project path in Ubuntu is now `~/reproducer`. No `/workspace` directory is
required. When you open another Ubuntu terminal, return and activate again:

```bash
cd ~/reproducer
source .venv/bin/activate
```

An optional integration check is:

```bash
CLM_TEST_DOCKER=1 python -m unittest discover -s tests -v
```

## 3. Configure model access

Open the configuration from Ubuntu:

```bash
nano configs/experiment.json
```

### Claude 4.6 Sonnet

Obtain an Anthropic API key with access to `claude-sonnet-4-6`. Enter it locally without
putting its value into a command, configuration file, or chat:

```bash
read -rsp "Anthropic API key: " ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY
```

### Qwen3.6-27B

You need an OpenAI-compatible endpoint serving the actual Qwen3.6-27B model. The experiment
package does not include model weights or start a model server. A typical laptop is not
an adequate full-precision 27B model server; use a suitable GPU server or hosted endpoint.

If you have a compatible vLLM installation on a GPU machine, a server command is:

```bash
vllm serve Qwen/Qwen3.6-27B --max-model-len 65536 --reasoning-parser qwen3
```

In your WSL experiment terminal, set the actual address of that server:

```bash
export QWEN_BASE_URL="http://YOUR-GPU-SERVER:8000/v1"
```

Replace `YOUR-GPU-SERVER` before running the command. `127.0.0.1` works only if the server
is reachable at that address from the WSL environment.

For an authenticated hosted endpoint, also set `require_key` to `true` in the `qwen27b`
configuration and enter its key locally:

```bash
read -rsp "Qwen endpoint API key: " QWEN_API_KEY
echo
export QWEN_API_KEY
```

Use the exact served model ID in the configuration. The default tokenizer endpoint is
vLLM's `/tokenize`. If your provider lacks it, choose `token_counter: "bytes"`; that changes
the context-budget measurement and must be disclosed in experiment results.

Check configuration presence:

```bash
python -m clm_experiment doctor --config configs/experiment.json
```

This prints presence checks, not credential values. It does not prove that a model endpoint
is reachable or that your account has access. These environment variables need to be set
again in a new terminal unless you manage them through your own secure environment setup.

## 4. Prepare math tasks

```bash
python -m clm_experiment prepare-math --out data/math --image clm-work:0.1
```

This creates `data/math/tasks.jsonl` and feasible initial constructions. These instances
are evaluation-only. They cannot be used to train the learned condition.

Use the offline fixtures to check setup. Keep the four math instances for their one
preregistered primary run per condition and model; do not run extra setup pilots on them.
The learned condition requires training data from the other benchmarks first.

## 5. Prepare BrowseComp-Plus

The real benchmark files are not included in this Git repository. Download them locally:

```bash
mkdir -p data
git clone https://github.com/texttron/BrowseComp-Plus.git ../BrowseComp-Plus
python -m pip install datasets
python ../BrowseComp-Plus/scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl
```

Export the public fixed corpus to JSONL. This downloads the corpus and can take substantial
time and disk space:

```bash
python - <<'PY'
import json
from datasets import load_dataset

rows = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", streaming=True)
with open("data/corpus.jsonl", "w", encoding="utf-8") as output:
    for row in rows:
        output.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
PY
```

Then index and split the queries:

```bash
python -m clm_experiment index-corpus --corpus data/corpus.jsonl --out data/corpus.db
python -m clm_experiment prepare-browse \
  --queries data/browsecomp_plus_decrypted.jsonl \
  --qrels ../BrowseComp-Plus/topics-qrels/qrel_evidence.txt \
  --corpus-db data/corpus.db --out data/browse
```

The resulting manifest has separate train and test queries. The built-in progress score is
a local exact-match/evidence proxy; official final grading is a separate upstream step
described in the README. Follow upstream access instructions if Hugging Face requests login.

## 6. Prepare EdgeBench

This integration requires task-specific setup rather than a ready-made task bundle. The
exact paper's EdgeBench-10 task list has not been selected automatically.

1. Install the SForge version matching your EdgeBench task definitions, fetch definitions,
   and pull the work and judge images for each chosen task. Start from
   [the official SForge instructions](https://github.com/ByteDance-Seed/EdgeBench).
2. Copy `configs/edge-selection.example.json` to `configs/edge-selection.local.json`.
3. Replace its example task JSON paths and image-tag placeholders with actual values.
   Set disjoint train/test groups and predeclare score milestones and targets.
4. Start `sforge serve` in another terminal. Configure the experiment terminal's
   `SFORGE_JUDGE_URL` and `SFORGE_ADMIN_SECRET` using that server's actual settings.
5. Import the workspaces:

   ```bash
   python -m clm_experiment prepare-edge \
     --selection configs/edge-selection.local.json --out data/edge
   ```

Example upstream commands for the two example task IDs are:

```bash
sforge fetch-tasks edgebench
sforge pull --task ad_placement_optimization --registry seededge
sforge pull --task integer_compression_codec --registry seededge
```

These commands alone do not fill in image tags, configure the judge, or define a full
EdgeBench-10 experiment. Consult the README for supported task restrictions. You can first
run BrowseComp-Plus and math without EdgeBench by omitting its `--tasks` argument below.

## 7. Train the two action policies

After the manifests and model endpoints are available:

```bash
python -m clm_experiment train --config configs/experiment.json \
  --tasks data/browse/tasks.jsonl --tasks data/edge/tasks.jsonl \
  --model qwen27b --episodes 32 --out policies/qwen27b
python -m clm_experiment train --config configs/experiment.json \
  --tasks data/browse/tasks.jsonl --tasks data/edge/tasks.jsonl \
  --model sonnet46 --episodes 32 --out policies/sonnet46
```

These are real model calls. Each backbone gets its own `policy.json`. Train on disjoint
training tasks, then freeze the configuration and implementation before evaluation.

## 8. Run and analyze the full experiment

```bash
python -m clm_experiment run --config configs/experiment.json \
  --tasks data/browse/tasks.jsonl --tasks data/math/tasks.jsonl \
  --tasks data/edge/tasks.jsonl --policy-dir policies --out runs/main

python -m clm_experiment analyze --runs runs/main --out runs/analysis.json
```

This schedules all three conditions for both configured backbones. There is one primary
run per math instance, method, and backbone; diagnostic continuation forks are additional
work and are charged separately. Default call budgets are development settings, not the
paper's 12-hour protocol.

Read [the validity protocol](protocol.md) before interpreting results. An inconclusive
analysis can be correct: independent task counts, artifact review, and usable checkpoints
are required for a confirmatory conclusion.

## Common errors

| Error | Action |
| --- | --- |
| `/workspace/reproducer: No such file or directory` | Clone locally, then use `~/reproducer`. |
| `py: command not found` | Install Windows Python with its launcher and reopen the terminal. |
| `.venv/bin/python: No such file` on native Windows | Use `./.venv/Scripts/python.exe`. |
| `No module named clm_experiment` | Run installation from the repository folder with the same Python interpreter. |
| Output directory already exists | Choose a new output folder; existing runs are protected from overwrite. |
| Missing `policy.json` | Train that model's policy first, or explicitly run only a control condition. |
| Cannot connect to Docker daemon | Start Docker Desktop and enable Ubuntu WSL integration. |
| Cannot connect to Qwen | Check the actual server URL, availability, model ID, and authentication. |

Downloaded data, API keys, virtual environments, results, and policies are local; they are
not automatically transferred between the cloud machine, Windows, WSL, or GitHub.
