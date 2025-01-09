import argparse
from precog.utils.config import config, to_string
import subprocess


parser = argparse.ArgumentParser()
config = config(parser)

start_command = ["pm2", "start", "--name", f"{config.neuron.name}", 
                    "python3 -m precog.validators.validator " + to_string(config)]

print(start_command)

subprocess.run(start_command)