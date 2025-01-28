import os

import bittensor as bt
import wandb

from precog import __version__, constants


def setup_wandb(self) -> None:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None:
        wandb.init(
            project=f"sn{self.config.netuid}-validators",
            entity=constants.WANDB_PROJECT,
            config={
                "hotkey": self.wallet.hotkey.ss58_address,
            },
            name=f"validator-{self.my_uid}-{__version__}",
            resume="auto",
            dir=self.config.neuron.full_path,
            reinit=True,
        )
    else:
        bt.logging.error("WANDB_API_KEY not found in environment variables.")


def log_wandb(responses, rewards, miner_uids):
    wandb_val_log = {
        "miners_info": {
            **{
                miner_uid: {
                    # Convert predictions array to a dict with named dimensions
                    **{f"miner_prediction_{i}": pred_val for i, pred_val in enumerate(response.prediction)},
                    # Convert interval array to lower/upper bounds
                    "interval_lower": response.interval[0],
                    "interval_upper": response.interval[1],
                    "miner_reward": reward,
                }
                for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
            },
        }
    }
    wandb.log(wandb_val_log)
