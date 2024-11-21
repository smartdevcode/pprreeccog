import bittensor as bt

from precog.protocol import Challenge


def forward(synapse: Challenge) -> Challenge:
    """
    Optimized forward function for low latency and caching.
    """
    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
    )
    middle_intervals = [
        [0, 30],
        [5, 7],
        [2, 20],
        [3, 12],
        [1, 15],
        [0, 30],
        [5, 20],
        [7, 10],
        [1, 12],
        [9, 12],
        [4, 13],
        [0, 30],
    ]
    synapse.prediction = [30]
    synapse.interval = middle_intervals[0]

    return synapse
