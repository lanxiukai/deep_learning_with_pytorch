import random
import torch


def set_seed(seed: int | None = None) -> None:
    # Use a high-entropy / high-resolution seed when not provided.
    # NOTE: int(time.time()) has only 1s resolution and can collide easily.
    if seed is None:
        seed = int(torch.seed())
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is not strictly required; RBM uses randomness anyway.
    # If you want more determinism, uncomment:
    # torch.use_deterministic_algorithms(True)
