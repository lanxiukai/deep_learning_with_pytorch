# dl_utils dependency-tree documentation

`dl_utils` is the repository's shared utility package for the PyTorch lessons.
Install it from the repository root after creating the Conda environment:

```bash
pip install -e .
```

For a pip-only setup that also includes dependencies used directly by lesson
and tool scripts, install the `examples` extra:

```bash
pip install -e ".[examples]"
```

**Reading conventions**

- `→` denotes a direct import dependency; `.` and `..` denote package-relative imports.
- Dependencies are listed in source import order. Declarations and class methods are listed in source order.
- Standard-library, third-party, and `dl_utils` dependencies appear together; resolve relative imports from the owning module's package.
- Names prefixed with `_` are internal implementation details. Treat unprefixed names and symbols listed in `__all__` as public-API candidates.
- Download, directory-reset, artifact-writing, and random-seed operations have external side effects; inspect their owning module before calling them.

## Section 1: Directory and module tree (alphabetical)

```
dl_utils/
├── __init__.py
│
├── d2l/
│   ├── __init__.py
│   ├── attention.py
│   ├── benchmark.py
│   ├── cnn.py
│   ├── data_fashion.py
│   ├── linear.py
│   ├── optim.py
│   ├── rnn.py
│   ├── seq2seq.py
│   ├── time_machine.py
│   ├── translation.py
│   └── vocabulary.py
│
├── data/
│   ├── __init__.py
│   ├── dataset_preparation.py
│   ├── downloads.py
│   ├── glasses_label_corrections.json
│   └── vision.py
│
├── devices/
│   ├── __init__.py
│   ├── randomness.py
│   └── selection.py
│
├── ebm/
│   ├── __init__.py
│   ├── _ebm_types.py
│   ├── dbm_diagnostics.py
│   ├── dbm_model.py
│   ├── dbm_sampling.py
│   ├── dbm_training.py
│   ├── dbn.py
│   ├── rbm_artifacts.py
│   ├── rbm_model.py
│   ├── rbm_primitives.py
│   ├── rbm_sampling.py
│   ├── rbm_training.py
│   ├── rbm_update.py
│   ├── sampling_artifacts.py
│   ├── sampling_common.py
│   ├── training_artifacts.py
│   └── training_curves.py
│
├── filesystem/
│   ├── __init__.py
│   ├── directories.py
│   └── project_root.py
│
├── genai/
│   ├── __init__.py
│   ├── cyclegan.py
│   ├── ddpm.py
│   ├── gan.py
│   ├── pix2pix.py
│   ├── sagan_biggan.py
│   ├── stylegan2.py
│   └── vae.py
│
├── plot/
│   ├── __init__.py
│   ├── _backend.py
│   ├── figures.py
│   └── images.py
│
└── training/
    ├── __init__.py
    ├── metrics.py
    ├── optimization.py
    └── timing.py
```

## Section 2: Per-module entry reference

### dl_utils/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/d2l/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/d2l/attention.py

Dependencies: math, torch, torch.nn (as nn), dl_utils.d2l.seq2seq (Decoder, Encoder, sequence_mask).
Public entries (__all__): masked_softmax, AdditiveAttention, DotProductAttention, AttentionDecoder, MultiHeadAttention, transpose_qkv, transpose_output, PositionalEncoding, PositionWiseFFN, AddNorm, EncoderBlock, TransformerEncoder.

### dl_utils/d2l/benchmark.py

Dependencies: torch.nn, dl_utils.d2l.cnn (Residual), dl_utils.training.timing (Timer).
Public entries: Benchmark, split_batch, resnet18.

### dl_utils/d2l/cnn.py

Dependencies: torch, torch.nn, torch.nn.functional, dl_utils.plot.figures (Animator), dl_utils.training.metrics (Accumulator, accuracy, evaluate_accuracy_gpu), dl_utils.training.timing (Timer).
Public entries: corr2d, train_ch6, Residual.

### dl_utils/d2l/data_fashion.py

Dependencies: torch, dl_utils.data.vision (vision_loaders), dl_utils.filesystem.project_root (infer_project_root), dl_utils.plot.figures (Animator), dl_utils.plot.images (show_images), dl_utils.training.metrics (Accumulator, accuracy, evaluate_accuracy), dl_utils.training.timing (Timer).
Public entries: get_fashion_mnist_labels, load_data_fashion_mnist, train_epoch_ch3, train_ch3, predict_ch3.

### dl_utils/d2l/linear.py

Dependencies: torch.
Public entries: synthetic_data, linreg, squared_loss.

### dl_utils/d2l/optim.py

Dependencies: numpy, torch, torch.nn, dl_utils.data.downloads (download), dl_utils.data.vision (load_array), dl_utils.d2l.linear (linreg, squared_loss), dl_utils.plot.figures (Animator), dl_utils.training.metrics (evaluate_loss), dl_utils.training.timing (Timer).
Public entries: train_2d, get_data_ch11, train_ch11, train_concise_ch11.

### dl_utils/d2l/rnn.py

Dependencies: math, torch, torch.nn, torch.nn.functional, dl_utils.plot.figures (Animator), dl_utils.training.metrics (Accumulator), dl_utils.training.optimization (grad_clipping, sgd), dl_utils.training.timing (Timer).
Public entries: RNNModelScratch, predict_ch8, train_epoch_ch8, train_ch8, RNNModel.

### dl_utils/d2l/seq2seq.py

Dependencies: collections, math, torch, torch.nn, dl_utils.d2l.translation (truncate_pad), dl_utils.plot.figures (Animator), dl_utils.training.metrics (Accumulator), dl_utils.training.optimization (grad_clipping), dl_utils.training.timing (Timer).
Public entries: Encoder, Decoder, EncoderDecoder, Seq2SeqEncoder, sequence_mask, MaskedSoftmaxCELoss, train_seq2seq, predict_seq2seq, bleu.

### dl_utils/d2l/time_machine.py

Dependencies: random, re, torch, dl_utils.data.downloads (download), .vocabulary (Vocab, tokenize).
Public entries: read_time_machine, load_corpus_time_machine, seq_data_iter_random, seq_data_iter_sequential, SeqDataLoader, load_data_time_machine.

### dl_utils/d2l/translation.py

Dependencies: os, torch, torch.utils.data, dl_utils.data.downloads (download_extract), .vocabulary (Vocab).
Public entries: read_data_nmt, preprocess_nmt, tokenize_nmt, truncate_pad, build_array_nmt, load_data_nmt.

### dl_utils/d2l/vocabulary.py

Dependencies: collections.
Public entries: tokenize, Vocab, count_corpus.

### dl_utils/data/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/data/downloads.py

Dependencies: hashlib, hmac, os, shutil, tarfile, tempfile, zipfile, urllib.parse (urlsplit), requests, dl_utils.filesystem.project_root (infer_project_root).
Public entries: DATA_HUB, DATA_URL, DOWNLOAD_TIMEOUT, DOWNLOAD_CHUNK_SIZE, download, download_extract, download_all.
Downloads require HTTPS, verify the upstream D2L SHA-1 after transfer, and atomically replace cache files only after validation succeeds.
`download`, `download_extract`, and `download_all` accept `data_root` to apply
the registry's standard subdirectory layout below a caller-selected root;
`cache_dir` remains available when an exact cache directory is required.

### dl_utils/data/dataset_preparation.py

Dependencies: csv, filecmp, json, os, shutil, concurrent.futures
(ThreadPoolExecutor, as_completed), pathlib (Path), PIL (Image), and optional
kagglehub inside `download_kaggle_dataset`.
Public entries: download_kaggle_dataset, prepare_celeba_cyclegan_splits,
resize_image, build_image_folder_cache, load_corrections,
ensure_glasses_classification, validate_glasses_classification,
apply_glasses_label_corrections.

### dl_utils/data/vision.py

Dependencies: os, pathlib (Path), typing (TypeAlias), torch, torch.utils.data, torchvision, torchvision.transforms.
Public entries: TensorBatch, TensorDataLoader, vision_loaders, image_folder_dataset, image_folder_loader, load_array.

### dl_utils/devices/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/devices/randomness.py

Dependencies: random, torch.
Public entries: set_seed.

### dl_utils/devices/selection.py

Dependencies: torch, typing (Optional, Union).
Public entries: get_device, try_gpu, try_all_gpus.

### dl_utils/ebm/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/ebm/_ebm_types.py

Dependencies: copy, os, dataclasses (dataclass), typing (NotRequired, Protocol, Sequence, TypeAlias, TypedDict), ..training.metrics (NumericScalar).
Public entries: RBMUpdateMetrics, DBMFineTuneMetrics, MCMCMetrics, SamplingOutputDirs, MetricHistory, StepMetrics, EpochMetrics, DBMPretrainMetrics, Config, LayerConfig, layer_cfg.

### dl_utils/ebm/dbm_diagnostics.py

Dependencies: math, os, torch, matplotlib.pyplot, ._ebm_types (MCMCMetrics).
Public entries: none.

### dl_utils/ebm/dbm_model.py

Dependencies: os, torch, torch.nn, ._ebm_types (DBMFineTuneMetrics).
Public entries: DBM.

### dl_utils/ebm/dbm_sampling.py

Dependencies: math, os, imageio.v2, imageio.v3, torch, imageio.typing (ArrayLike), tqdm.auto (tqdm), ..data.vision (vision_loaders), ..devices.randomness (set_seed), ..filesystem.directories (reset_dir), ..plot.images (save_grid), ..training.timing (Timer), ._ebm_types (_DBMConfig), .dbm_diagnostics (_compute_mcmc_metrics, _save_mcmc_convergence_plots), .dbm_model (DBM), .rbm_model (binarize), .sampling_artifacts (ensure_dir), .sampling_common (_desync_prepare, _parse_save_steps).
Public entries: sampling_dbm.

### dl_utils/ebm/dbm_training.py

Dependencies: math, os, collections.abc (Callable), torch, torch.nn.functional, tqdm.auto (tqdm), ..data.vision (TensorDataLoader, vision_loaders), ..devices.randomness (set_seed), ..filesystem.directories (reset_dir), ..plot.images (save_grid), ..training.metrics (NumericScalar, save_metrics_csv), ..training.timing (Timer), ._ebm_types (DBMPretrainMetrics, EpochMetrics, MetricHistory, StepMetrics, _DBMConfig, layer_cfg), .dbm_model (DBM), .rbm_model (BinaryRBM, binarize, rbm_energy_free_energy_fast), .training_artifacts (save_rbm_training_artifacts).
Public entries: train_dbm.

### dl_utils/ebm/dbn.py

Dependencies: math, os, torch, tqdm.auto (tqdm), ..devices.randomness (set_seed), ..plot.images (save_grid), ..training.timing (Timer), ._ebm_types (LayerConfig, _DBNRunConfig, _DBNSamplingConfig, layer_cfg), .rbm_model (BinaryRBM, binarize, rbm_energy_free_energy_fast), .rbm_training (_configure_rbm_fixed_point, _prepare_layer1_rbm, init_hidden_visible_bias, train_rbm_layer), .sampling_artifacts (save_gibbs_energy_curves, save_gif_from_grids, save_sample_data, save_timelines_for_samples), .sampling_common (_desync_gibbs_visible_chains, _infer_gen_desync_steps, _parse_save_steps, _sampling_output_dirs, checkpoint_path, load_rbm_model_from_ckpt, prepare_init_visible).
Public entries: train_dbn, sampling_dbn.

### dl_utils/ebm/rbm_artifacts.py

Dependencies: math, os, torch, ..plot.images (save_grid), dl_utils.ebm._ebm_types (_RBMArtifactConfig), dl_utils.ebm.rbm_model (BinaryRBM).
Public entries: save_filters_grids, save_rbm_recon_grids.

### dl_utils/ebm/rbm_model.py

Dependencies: typing (Tuple), torch, torch.nn, torch.nn.functional, ._ebm_types (RBMUpdateMetrics), .rbm_primitives (binarize, _quantize_int16_q_, rbm_energy_free_energy_fast), .rbm_update (cd_k_update as _cd_k_update_fn).
Public entries: BinaryRBM.

### dl_utils/ebm/rbm_primitives.py

Dependencies: __future__ (annotations), typing (TYPE_CHECKING), torch, torch.nn.functional, .rbm_model (BinaryRBM — TYPE_CHECKING only).
Public entries: binarize, rbm_energy_free_energy_fast.

### dl_utils/ebm/rbm_sampling.py

Dependencies: math, os, torch, tqdm.auto (tqdm), ..devices.randomness (set_seed), ..plot.images (save_grid), ..training.timing (Timer), ._ebm_types (_RBMSamplingConfig), .rbm_model (rbm_energy_free_energy_fast), .sampling_artifacts (save_gibbs_energy_curves, save_gif_from_grids, save_sample_data, save_timelines_for_samples), .sampling_common (_desync_gibbs_visible_chains, _infer_gen_desync_steps, _parse_save_steps, _sampling_output_dirs, load_rbm_model_from_ckpt, prepare_init_visible, validate_out_dir_and_ckpt).
Public entries: sampling_rbm.

### dl_utils/ebm/rbm_training.py

Dependencies: math, collections.abc (Callable), torch, torch.nn.functional, tqdm.auto (tqdm), ..data.vision (TensorDataLoader, vision_loaders), ..devices.randomness (set_seed), ..filesystem.directories (reset_dir), ..training.timing (Timer), ._ebm_types (EpochMetrics, StepMetrics, _RBMFixedPointConfig, _RBMRunConfig, _RBMSetupConfig, _RBMTrainingConfig), .rbm_artifacts (save_filters_grids, save_rbm_recon_grids), .rbm_model (BinaryRBM, binarize, rbm_energy_free_energy_fast), .training_artifacts (save_rbm_training_artifacts).
Public entries: init_bv_from_data, init_hidden_visible_bias, train_rbm_layer, train_rbm.

### dl_utils/ebm/rbm_update.py

Dependencies: __future__ (annotations), math, typing (TYPE_CHECKING), torch, torch.nn.functional, ._ebm_types (RBMUpdateMetrics), .rbm_primitives (rbm_energy_free_energy_fast), .rbm_model (BinaryRBM — TYPE_CHECKING only).
Public entries: cd_k_update.

### dl_utils/ebm/sampling_artifacts.py

Dependencies: os, collections.abc (Sequence), imageio.v2, torch, matplotlib.pyplot, ..plot.figures (save_curve), dl_utils.ebm._ebm_types (_DBNSamplingConfig, _RBMSamplingConfig).
Public entries: ensure_dir, save_sample_data, save_timelines_for_samples, save_gif_from_grids, save_gibbs_energy_curves.

### dl_utils/ebm/sampling_common.py

Dependencies: os, collections.abc (Sized), torch, ..data.vision (TensorDataLoader, vision_loaders), ..filesystem.directories (reset_dir), dl_utils.ebm._ebm_types (SamplingOutputDirs, _DBMConfig, _DBNSamplingConfig, _InitialVisibleConfig, _RBMFixedPointConfig, _RBMLoadConfig, _RBMSamplingConfig, _SamplingOutputConfig), dl_utils.ebm.rbm_model (BinaryRBM, binarize), dl_utils.ebm.sampling_artifacts (ensure_dir).
Public entries: checkpoint_path, validate_out_dir_and_ckpt, load_rbm_model_from_ckpt, prepare_init_visible.

### dl_utils/ebm/training_artifacts.py

Dependencies: os, torch, ..data.vision (TensorDataLoader), ..training.metrics (align_and_drop_all_nan_rows, align_metrics_for_csv, save_metrics_csv), dl_utils.ebm._ebm_types (EpochMetrics, StepMetrics, _RBMArtifactConfig), dl_utils.ebm.rbm_model (BinaryRBM), .training_curves (save_metric_curves).
Public entries: save_rbm_training_artifacts.

### dl_utils/ebm/training_curves.py

Dependencies: os, ..plot.figures (maybe_save_curve), ..training.metrics (as_list), dl_utils.ebm._ebm_types (EpochMetrics, StepMetrics).
Public entries: save_metric_curves.

### dl_utils/filesystem/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/filesystem/directories.py

Dependencies: shutil, os, .project_root (infer_project_root).
Public entries: clean_pycache, reset_dir.

### dl_utils/filesystem/project_root.py

Dependencies: pathlib (Path).
Public entries: infer_project_root.
Runnable project scripts use this helper to anchor `data/` and `output/` paths independently of the caller's working directory.

### dl_utils/genai/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker, documents the generative-model submodules).

### dl_utils/genai/cyclegan.py

Dependencies: os, torch, numpy, torch.nn, PIL.Image, torch.utils.data (Dataset), tqdm, torchvision.utils (save_image).
Public entries: test, train_epoch, ConvBlock, ResidualBlock, Generator, Block, Discriminator, LoadData, weights_init.

### dl_utils/genai/pix2pix.py

Dependencies: csv, pathlib, albumentations, numpy, torch, PIL, torch.utils.data (Dataset), dl_utils.data.images (load_rgb_image).
Public entries: build_paired_transform, CelebAColorizationDataset, DownBlock, UpBlock, UNetGenerator, ConditionalPatchDiscriminator, initialize_weights, denormalize.

### dl_utils/genai/sagan_biggan.py

Dependencies: torch, torch.nn.functional, torch (nn), torch.nn.utils (spectral_norm).
Public entries: SelfAttention, ConditionalBatchNorm2d, GeneratorResidualBlock, ConditionalGenerator, DiscriminatorResidualBlock, ProjectionDiscriminator, truncated_normal, denormalize.

### dl_utils/genai/stylegan2.py

Dependencies: math, random, torch, torch.nn.functional, torch (nn).
Public entries: PixelNorm, EqualizedLinear, MappingNetwork, ModulatedConv2d, NoiseInjection, StyledConv, ToRGB, SynthesisBlock, StyleGenerator, DiscriminatorResidualBlock, MinibatchStandardDeviation, StyleDiscriminator, denormalize.

### dl_utils/genai/ddpm.py

Dependencies: math, typing (Union), numpy, torch, einops (rearrange), einops.layers.torch (Rearrange), torch (einsum, nn), torchvision.transforms (CenterCrop, Compose, InterpolationMode, RandomHorizontalFlip, Resize, ToTensor), tqdm.
Public entries (__all__): transforms, DDIMScheduler, UNet.

### dl_utils/genai/gan.py

Dependencies: torch, torch.nn (as nn).
Public entries: gradient_penalty, update_D, update_G, Generator, Critic.

### dl_utils/genai/vae.py

Dependencies: torch, torch.nn.functional, torch.nn, dl_utils.devices.selection (get_device).
Public entries: device, VAEEncoder, VAEDecoder, VAE.

### dl_utils/plot/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/plot/_backend.py

Dependencies: os, matplotlib.
Public entries: none.

### dl_utils/plot/figures.py

Dependencies: csv, os, collections.abc (Mapping, Sequence), os (PathLike), typing (Any), dl_utils.plot._backend, matplotlib_inline.backend_inline, matplotlib.pyplot, dl_utils.training.metrics (MetricHistory, as_list, has_any_finite).
Public entries: use_svg_display, set_figsize, set_axes, plot, annotate, Animator, heatmap, trace2d, seq_len_hist, save_curve, maybe_save_curve.

### dl_utils/plot/images.py

Dependencies: math, os, pathlib (Path), typing (Any), dl_utils.plot._backend, numpy, torch, torchvision, matplotlib.pyplot.
Public entries: show_images, save_grid, vae_sample_grid.

### dl_utils/training/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/training/metrics.py

Dependencies: csv, math, os, collections.abc (Mapping, Sequence), os (PathLike), typing (TypeAlias), torch, torch.nn.
Public entries: NumericScalar, MetricHistory, Accumulator, accuracy, evaluate_accuracy, evaluate_accuracy_gpu, evaluate_loss, as_list, has_any_finite, align_metrics_for_csv, align_and_drop_all_nan_rows, save_metrics_csv.

### dl_utils/training/optimization.py

Dependencies: torch, torch.nn.
Public entries: sgd, grad_clipping.

### dl_utils/training/timing.py

Dependencies: time, numpy.
Public entries: Timer, format_epoch_timing.
