# dl_utils dependency-tree documentation

`dl_utils` is the repository's shared utility package for the PyTorch lessons.
Install it from the repository root through the project-local uv environment:

```bash
uv sync --locked
```

For a pip-only setup that also includes dependencies used directly by lesson
and tool scripts, install the `examples` extra:

```bash
uv sync --all-extras --locked
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
│   ├── celeba.py
│   ├── dataset_preparation.py
│   ├── downloads.py
│   ├── glasses_label_corrections.json
│   ├── images.py
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
│   ├── biggan.py
│   ├── cyclegan.py
│   ├── ddpm.py
│   ├── diffusion_ddpm.py
│   ├── diffusion_score_sde.py
│   ├── diffusion_unet.py
│   ├── gan.py
│   ├── pix2pix.py
│   ├── perceptual_autoencoder.py
│   ├── progan.py
│   ├── quantization.py
│   ├── sagan.py
│   ├── sn_gan.py
│   ├── stylegan.py
│   ├── stylegan2.py
│   ├── stylegan_common.py
│   ├── token_prior.py
│   ├── vae.py
│   ├── vae_common.py
│   └── vae_hierarchy.py
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
    ├── parameters.py
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

### dl_utils/data/celeba.py

Dependencies: __future__ (annotations), csv, pathlib (Path),
torch.utils.data (Dataset), dl_utils.data.images (load_rgb_image).
Public entries (__all__): CELEBA_ALIGNED_CROP_SIZE, CELEBA_PARTITIONS,
CelebAAlignedDataset.
The dataset reads the official split manifest from the locally prepared
aligned CelebA tree and returns unconditional `(image, 0)` samples.

### dl_utils/data/downloads.py

Dependencies: hashlib, hmac, os, shutil, tarfile, tempfile, zipfile, urllib.parse (urlsplit), requests, dl_utils.filesystem.project_root (infer_project_root).
Public entries: DATA_HUB, DATA_URL, DOWNLOAD_TIMEOUT, DOWNLOAD_CHUNK_SIZE, download, download_extract, download_all.
Downloads require HTTPS, verify the upstream D2L SHA-1 after transfer, and atomically replace cache files only after validation succeeds.
`download`, `download_extract`, and `download_all` accept `data_root` to apply
the registry's standard subdirectory layout below a caller-selected root;
`cache_dir` remains available when an exact cache directory is required.

### dl_utils/data/dataset_preparation.py

Dependencies: csv, filecmp, json, os, shutil, concurrent.futures
(ThreadPoolExecutor, as_completed), pathlib (Path), PIL (Image),
dl_utils.data.images (flatten_to_rgb), and optional kagglehub inside
`download_kaggle_dataset`.
Public entries: download_kaggle_dataset, prepare_celeba_cyclegan_splits,
resize_image, build_image_folder_cache, load_corrections,
ensure_glasses_classification, validate_glasses_classification,
apply_glasses_label_corrections.

### dl_utils/data/images.py

Dependencies: pathlib (Path), PIL (Image).
Public entries: flatten_to_rgb, load_rgb_image.
Both helpers preserve palette and explicit alpha information by compositing
transparent images onto a caller-selectable solid background before returning
RGB data.

### dl_utils/data/vision.py

Dependencies: os, pathlib (Path), typing (TypeAlias), torch, torch.utils.data,
torchvision, torchvision.transforms, dl_utils.data.images (load_rgb_image).
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

### dl_utils/genai/biggan.py

Dependencies: torch, torch.nn.functional (as F), torch (nn), torch.nn.utils
(spectral_norm), dl_utils.genai.sagan (SAGANDiscriminator, SelfAttention).
Public entries: ConditionalBatchNorm2d, BigGANGeneratorResidualBlock,
CompactBigGANGenerator, BigGANDiscriminator,
modified_orthogonal_regularization, truncated_normal.

### dl_utils/genai/cyclegan.py

Dependencies: os, matplotlib.pyplot, torch, numpy, torch.nn, torch.utils.data
(Dataset), tqdm, dl_utils.data.images (load_rgb_image).
Public entries: LAMBDA_CYCLE, save_translation_snapshot, train_epoch, ConvBlock,
ResidualBlock, Generator, Block, Discriminator, LoadData, weights_init.

### dl_utils/genai/ddpm.py

Dependencies: dl_utils.genai.diffusion_ddpm, diffusion_score_sde,
diffusion_unet.
Public entries (__all__): DiffusionPrediction, DiffusionUNet, DiscreteSampler,
GaussianDiffusion, PredictionType, ScoreSampler, UNet, VPSDE,
cosine_beta_schedule, linear_beta_schedule, sample_vp_sde. This module is a
compatibility facade; new lessons import the focused modules directly.

### dl_utils/genai/diffusion_ddpm.py

Dependencies: math, dataclasses, typing, torch, torch.nn.functional.
Public entries (__all__): DiffusionPrediction, DiscreteSampler,
GaussianDiffusion, PredictionType, cosine_beta_schedule,
linear_beta_schedule.

### dl_utils/genai/diffusion_score_sde.py

Dependencies: dataclasses, typing, torch.
Public entries (__all__): ScoreSampler, VPSDE, sample_vp_sde.

### dl_utils/genai/diffusion_unet.py

Dependencies: math, typing, torch, torch.nn.functional.
Public entries (__all__): DiffusionUNet, UNet.

### dl_utils/genai/gan.py

Dependencies: torch, torch.nn (as nn).
Public entries: gradient_penalty, update_D, update_G, Generator, Critic.

### dl_utils/genai/pix2pix.py

Dependencies: __future__ (annotations), csv, pathlib (Path), albumentations,
albumentations.pytorch (ToTensorV2), numpy, torch, PIL (ImageOps), torch
(Tensor, nn), torch.utils.data (Dataset), dl_utils.data.images
(load_rgb_image).
Public entries: build_paired_transform, CelebAColorizationDataset, DownBlock,
UpBlock, UNetGenerator, ConditionalPatchDiscriminator, initialize_weights,
denormalize.

### dl_utils/genai/sagan.py

Dependencies: math, torch, torch (nn), torch.nn.utils (spectral_norm),
dl_utils.genai.sn_gan (ProjectionSNDiscriminator, SNGenerator).
Public entries: SelfAttention, SAGANGenerator, SAGANDiscriminator.

### dl_utils/genai/sn_gan.py

Dependencies: math, torch, torch.nn.functional (as F), torch (nn),
torch.nn.utils (spectral_norm).
Public entries: SNGeneratorResidualBlock, SNDiscriminatorResidualBlock,
SNGenerator, SNDiscriminator, count_spectral_norm_layers,
uniform_dequantize_uint8, discriminator_hinge_loss, generator_hinge_loss,
init_spectral_norm_state, spectral_norm_scratch_minimal.

### dl_utils/genai/stylegan_common.py

Dependencies: math, collections.abc (Sequence), torch,
torch.nn.functional (as F), torch (nn).
Public entries: RESOLUTIONS, CHANNEL_MULTIPLIERS, NOISE_MODES,
make_channel_map, validate_resolution, validate_alpha, PixelNorm,
EqualizedLinear, EqualizedConv2d, MinibatchStandardDeviation, NoiseInjection,
filter2d, filtered_upsample2d, filtered_downsample2d, denormalize.

### dl_utils/genai/progan.py

Dependencies: math, torch, torch.nn.functional (as F), torch (nn),
dl_utils.genai.stylegan_common.
Public entries: GeneratorInputBlock, GeneratorBlock, ProGANGenerator,
DiscriminatorBlock, ProGANDiscriminator, denormalize.

### dl_utils/genai/stylegan.py

Dependencies: math, random, torch, torch.nn.functional (as F), torch (nn),
dl_utils.genai.stylegan_common.
Public entries: MappingNetwork, AdaptiveInstanceNorm, StyledActivation,
SynthesisBlock, StyleGANGenerator, DiscriminatorBlock,
StyleGANDiscriminator, denormalize.

### dl_utils/genai/stylegan2.py

Dependencies: math, random, torch, torch.nn.functional (as F), torch (nn),
dl_utils.genai.stylegan_common.
Public entries: MappingNetwork, ModulatedConv2d, StyledConv, ToRGB,
SynthesisBlock, StyleGenerator, DiscriminatorResidualBlock,
StyleDiscriminator, denormalize.

### dl_utils/genai/vae.py

Dependencies: torch, torch.nn.functional, torch.nn, dl_utils.devices.selection (get_device).
Public entries: device, diagonal_gaussian_kl, reparameterize, VAEEncoder,
VAEDecoder, VAE.

### dl_utils/genai/vae_common.py

Dependencies: math, torch, torch.nn.
Public entries (__all__): LOG_2PI, split_gaussian_parameters,
reparameterize_logvar, diagonal_gaussian_kl_from_logvar,
diagonal_gaussian_log_density, fuse_diagonal_gaussians, ConvGaussianVAE28.

### dl_utils/genai/vae_hierarchy.py

Dependencies: torch, torch.nn.functional, torch.nn,
dl_utils.genai.vae_common.
Public entries (__all__): CompactHierarchicalVAE, ActiveUnitAccumulator,
hierarchical_vae_loss, active_units.

### dl_utils/genai/quantization.py

Dependencies: math, collections.abc (Sequence), torch,
torch.nn.functional, torch.nn.
Public entries (__all__): token_usage, TokenUsageAccumulator, VectorQuantizer,
FiniteScalarQuantizer, ResidualBlock, ImageEncoder32, ImageDecoder32,
VQVAE32, FSQAutoencoder32.

### dl_utils/genai/token_prior.py

Dependencies: math, torch, torch.nn.functional, torch.nn.
Public entries (__all__): MaskedConv2d, PixelCNNPrior,
CausalTransformerPrior.

### dl_utils/genai/perceptual_autoencoder.py

Dependencies: torch, torch.nn.functional, torch.nn,
dl_utils.genai.quantization, dl_utils.genai.vae_common;
torchvision.models is imported lazily only for pretrained VGG features.
Public entries (__all__): ResidualBlock, PerceptualEncoder32,
PerceptualDecoder32, VQPerceptualAutoencoder32,
KLPerceptualAutoencoder32, PatchDiscriminator32,
RandomFeaturePerceptualLoss, VGGPerceptualLoss,
discriminator_hinge_loss, adaptive_adversarial_weight.

### dl_utils/plot/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/plot/_backend.py

Dependencies: os, matplotlib.
Public entries: none.

### dl_utils/plot/figures.py

Dependencies: csv, os, collections.abc (Mapping, Sequence), os (PathLike), typing (Any), dl_utils.plot._backend, matplotlib_inline.backend_inline, matplotlib.pyplot, dl_utils.training.metrics (MetricHistory, as_list, has_any_finite).
Public entries: use_svg_display, set_figsize, set_axes, plot, annotate, Animator, heatmap, trace2d, seq_len_hist, save_curve, save_loss_curves, save_loss_panels, maybe_save_curve.
`save_loss_curves` writes total discriminator loss, total generator loss,
generator adversarial components, and weighted reconstruction components as
four vertically stacked plots with independent y-axes.
`save_loss_panels` accepts an ordered mapping of arbitrary loss groups and
writes each group on its own independent-y subplot.

### dl_utils/plot/images.py

Dependencies: math, os, pathlib (Path), typing (Any), dl_utils.plot._backend, numpy, torch, torchvision, matplotlib.pyplot.
Public entries: show_images, save_grid, save_image_row_grid,
save_fixed_noise_samples, save_training_samples, vae_sample_grid.
`save_image_row_grid` supports an optional figure title, column labels, and
output DPI while retaining labelled image rows.
`save_fixed_noise_samples` preserves an unconditional generator's mode while
rendering a fixed latent batch as labelled rows.
`save_training_samples` preserves the generator's current training mode while
rendering fixed class-conditional samples as labelled image rows. Its optional
AMP settings apply autocast to sample inference and convert the result back to
FP32 before rendering.

### dl_utils/training/__init__.py

Dependencies: none.
Public entries: none (docstring-only package marker).

### dl_utils/training/checkpoints.py

Dependencies: random, tempfile, collections.abc (Mapping), os (PathLike),
pathlib (Path), typing (Any), numpy, torch, torch (nn), torch.optim
(Optimizer).
Public entries (__all__): CHECKPOINT_FORMAT_VERSION, atomic_torch_save,
capture_rng_state, load_training_checkpoint, make_training_checkpoint,
restore_rng_state, save_model_weights, save_periodic_checkpoint.
The helpers atomically save latest/archive checkpoints, restore named models,
optimizers and RNG streams, and save metadata-rich final model weights.

### dl_utils/training/session.py

Dependencies: collections.abc (Mapping), os (PathLike), pathlib (Path),
typing (Any), torch (nn), torch.optim (Optimizer),
dl_utils.filesystem.directories (reset_dir),
dl_utils.training.checkpoints.
Public entries (__all__): TrainingSession.
`TrainingSession` manages fresh/resumed output lifecycles, periodic full-state
checkpoints, and final per-model weights without owning the training loop.

### dl_utils/training/metrics.py

Dependencies: csv, math, os, collections.abc (Mapping, Sequence), os (PathLike), typing (TypeAlias), torch, torch.nn.
Public entries: NumericScalar, MetricHistory, Accumulator, accuracy, evaluate_accuracy, evaluate_accuracy_gpu, evaluate_loss, as_list, has_any_finite, align_metrics_for_csv, align_and_drop_all_nan_rows, save_metrics_csv.

### dl_utils/training/optimization.py

Dependencies: torch, torch.nn.
Public entries: sgd, grad_clipping.

### dl_utils/training/parameters.py

Dependencies: torch.nn.
Public entries (__all__): count_parameters.
`count_parameters` counts all model parameters by default, excludes buffers,
and accepts `trainable_only=True` to count only gradient-enabled parameters.

### dl_utils/training/timing.py

Dependencies: time, numpy.
Public entries: Timer, format_epoch_timing.
