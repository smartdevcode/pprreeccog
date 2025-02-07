import argparse
import asyncio
import logging
from pathlib import Path

logging.getLogger("bittensor").propagate = False
logging.getLogger("urllib3").setLevel(logging.WARNING)

from precog.utils.config import config  # noqa: E402
from precog.validators.weight_setter import weight_setter  # noqa: E402


class Validator:
    def __init__(self, config):
        self.config = config
        full_path = Path(
            f"{self.config.logging.logging_dir}/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/validator"
        ).expanduser()
        full_path.mkdir(parents=True, exist_ok=True)
        self.config.full_path = str(full_path)

    async def main(self):
        loop = asyncio.get_event_loop()
        self.weight_setter = await weight_setter.create(config=self.config, loop=loop)
        # Add task manager
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Clean shutdown
            return

    async def reset_instance(self):
        self.__init__()
        asyncio.run(self.main())


# Run the validator.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = config(parser, neuron_type="validator")
    validator = Validator(config)
    asyncio.run(validator.main())
