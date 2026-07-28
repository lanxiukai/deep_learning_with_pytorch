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
│   ├── 1.0_variational_autoencoder/
│   ├── 2.0_generative_adversarial_network/
│   └── 3.0_diffusion_model/
│
├── tool_scripts/
│   ├── download_dataset_test.py
│   ├── plot_fashion_mnist.py
│   ├── pytorch_test.py
│   ├── sgd_animation.py
│   └── word_frequency.py
│
├── tests/                           # Test suite and one-off verification tools
│   └── batch_verify_glasses.py      # Historical Vision-LLM label audit
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
| CUDA | 13.0 |
| PyTorch | 2.9.1+cu130 |
| Python | 3.12 |

---

#### Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution
- Optional NVIDIA GPU: Turing or newer, with driver 580 or newer for CUDA 13.x
- Sufficient disk space for the environment and datasets

#### Setup

```bash
conda env create -f environment.yml
conda activate d2l
python -m pip install --no-deps -e .
```

`environment.yml` uses the CUDA 13.0 PyTorch wheels. For a CPU-only
environment, change `cu130` in the wheel index to `cpu` before creating it.

#### Dataset Preparation

To download the lesson datasets:

```bash
python tool_scripts/download_dataset_test.py
```

Some datasets are large, and Kaggle-hosted datasets may require authentication.
For the glasses dataset, the command resumes incomplete G/NoG classification,
applies the reviewed label corrections, and builds `data/glasses-256/`.
