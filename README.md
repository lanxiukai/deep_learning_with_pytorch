## Deep Learning with PyTorch

A Python code repository for introductory learning in deep learning and
generative AI, based on the reference books listed below and implemented with
the PyTorch framework.

#### Reference Books

- Aston Zhang et al. *Dive into Deep Learning* — https://d2l.ai
- Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning* — https://www.deeplearningbook.org
- Jakub M. Tomczak. *Deep Generative Modeling* — https://link.springer.com/book/10.1007/978-3-031-64087-2
- Mark Liu. *Learn Generative AI with PyTorch* — https://www.manning.com/books/learn-generative-ai-with-pytorch

---

#### Repository Structure

```
.
├── dl_utils/                        # Shared utility package; see dl_utils/README.md
│
├── deep_learning/                   # Deep learning lessons
│   ├── 0.0_neural_networks/
│   ├── 1.0_convolutional_neural_network/
│   ├── 2.0_recurrent_neural_network/
│   ├── 3.0_attention_mechanisms/
│   ├── 4.0_optimization/
│   └── 5.0_computational_performance/
│
├── genai/                           # Generative AI lessons
│   ├── 0.0_energy_based_model/
│   ├── 1.0_generative_adversarial_network/
│   ├── 2.0_variational_autoencoder/
│   └── 3.0_diffusion_model/
│
├── docs/                            # Implementation notes and model guides
├── environments/                    # Exact, checksummed runtime records
│   └── d2l/
│
├── tool_scripts/
│   ├── download_dataset_test.py
│   ├── plot_fashion_mnist.py
│   ├── pytorch_test.py
│   ├── sgd_animation.py
│   └── word_frequency.py
│
├── tests/                           # Utility and lesson regression tests
│
├── pyproject.toml
├── environment.yml
└── .gitignore
```

Lesson directories and files are numbered in suggested reading order; the
numbers do not correspond to book chapter numbers.

---

#### Experimental Environment

| Item | Version |
|---|---|
| OS | Ubuntu 24.04, x86_64 |
| GPU | NVIDIA GeForce RTX 4070 Ti |
| NVIDIA driver | 610.88 |
| PyTorch CUDA runtime | 13.0 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| Python | 3.14.6 |

---

#### Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution
- Optional NVIDIA GPU: Turing or newer, with driver 580 or newer for CUDA 13.x
- Sufficient disk space for the environment and datasets

#### Setup

```bash
conda env create -f environment.yml
conda activate d2l
python -m pip install --no-deps --no-build-isolation -e .
```

The human-maintained `environment.yml` is complemented by an exact,
checksummed runtime record under [`environments/d2l/`](environments/d2l/).
The record preserves the tested Conda artifacts and pip package set while
restoring this repository's `dl-utils` package separately as an editable
install.

These files have separate recovery roles and should be kept together:

- `environment.yml` is the readable dependency source for rebuilding forward;
- `environments/d2l/` is the exact, checksummed snapshot of the tested runtime;
- `pyproject.toml` defines the local `dl-utils` package and is required by the
  editable-install step above.

`environment.yml` installs PyTorch and all source-derived Python dependencies
with pip inside the Conda environment. The PyTorch and torchvision wheels come
from the official CUDA 13.0 (`cu130`) index and carry their environment-local
CUDA runtime dependencies; a system CUDA Toolkit is not required and the setup
does not modify the NVIDIA driver.

The dependency set covers imports in the active project source. Generated
`build/` copies and the ignored nested reference repositories under
`book_repos/` do not add packages to the environment. The repository uses the
local `dl_utils.d2l` package rather than the external PyPI `d2l` distribution.

For a CPU-only environment, change the wheel index suffix from `cu130` to `cpu`
and change the `torch` and `torchvision` version suffixes from `+cu130` to
`+cpu` before creating the environment. GPU lesson results will then differ.

#### Dataset Preparation

To download the lesson datasets:

```bash
python tool_scripts/download_dataset_test.py
```

Every destination in this script derives from its `DATA_DIR` constant. Change
that single constant to download and prepare the complete dataset collection
under a different root. Kaggle downloads also use that destination directly
instead of populating the default user-level Kaggle cache first.

Some datasets are large, and Kaggle-hosted datasets may require authentication.
For the glasses dataset, the command resumes incomplete G/NoG classification,
applies the reviewed label corrections, and builds `data/glasses-256/`.
