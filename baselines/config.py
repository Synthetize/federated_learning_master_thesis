"""Baselines configuration.

Everything comes from `federated.config.CONFIG`, no value is rewritten here. In the
previous version (baselines.py) alpha, seed, learning rate, momentum, epochs and minimum
partition size were copied by hand, with the comment "MUST match the sweep": forgetting
one was enough to compare the federated model with baselines trained under different
conditions, and the failure point would have been measured against the wrong reference.
Reading them from the same source makes that misalignment impossible.
"""

from __future__ import annotations

from federated.config import CONFIG, device, set_seeds   # noqa: F401  (set_seeds re-exported)

ALPHAS = list(CONFIG.alphas)
SEEDS = list(CONFIG.seeds)
NUM_CLIENTS = CONFIG.num_clients

# Epoch budget identical to what a client consumes inside the federation:
# num_rounds rounds x local_epochs local epochs.
EPOCHS = CONFIG.num_rounds * CONFIG.local_epochs
BATCH_SIZE = CONFIG.batch_size
LEARNING_RATE = CONFIG.learning_rate
MOMENTUM = CONFIG.momentum       # 0.0: with m=0.9 the low alphas die (see federated/config.py)

# Batch size for evaluation only: it does not affect the result, just the speed.
EVAL_BATCH_SIZE = 256

# The centralized model does not depend on alpha: merging the partitions rebuilds the same
# training pool. But get_fds needs an alpha to build the dataset, and the mildest value is
# the one the partitioner accepts at the first attempt. The result does not change.
CENTRAL_PARTITION_ALPHA = 10.0

DEVICE = device()
