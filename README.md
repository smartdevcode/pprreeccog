<div align="center">
<img src="docs/images/precog-logo.svg" />

# **CoinMetrics Precog Subnet** <!-- omit in toc -->

|     |     |
| :-: | :-: |
| **Status** | <img src="https://img.shields.io/github/v/release/coinmetrics/precog?label=Release" height="25"/> <img src="https://img.shields.io/github/actions/workflow/status/coinmetrics/precog/ci.yml?label=Build" height="25"/> <br> <a href="https://github.com/pre-commit/pre-commit" target="_blank"> <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&label=Pre-Commit" height="25"/> </a> <a href="https://github.com/psf/black" target="_blank"> <img src="https://img.shields.io/badge/code%20style-black-000000.svg?label=Code%20Style" height="25"/> </a> <br> <img src="https://img.shields.io/github/license/coinmetrics/precog?label=License" height="25"/> |
| **Activity** | <img src="https://img.shields.io/github/commit-activity/m/coinmetrics/precog?label=Commit%20Activity" height="25"/> <img src="https://img.shields.io/github/commits-since/coinmetrics/precog/latest/dev?label=Commits%20Since%20Latest%20Release" height="25"/> <br> <img src="https://img.shields.io/github/release-date/coinmetrics/precog?label=Latest%20Release%20Date" height="25"/> <img src="https://img.shields.io/github/last-commit/coinmetrics/precog/dev?label=Last%20Commit" height="25"/> <br> <img src="https://img.shields.io/github/contributors/coinmetrics/precog?label=Contributors" height="25"/> |
| **Compatibility** | <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcoinmetrics%2Fprecog%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.python&logo=python&label=Python&logoColor=yellow" height="25"/> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcoinmetrics%2Fprecog%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.bittensor&prefix=v&label=Bittensor" height="25"/> <br> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcoinmetrics%2Fprecog%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.coinmetrics-api-client&prefix=v&label=coinmetrics-api-client" height="25"/> |
| **Social** | <a href="" target="_blank"> <img src="https://img.shields.io/website?url=https%3A%2F%2Fcharts.coinmetrics.io%2Fcrypto-data%2F&up_message=CoinMetrics&label=Website" height="25"/> </a> |


</div>

---

- [Introduction](#introduction)
- [Design Decisions](#design-decisions)
- [Installation](#installation)
- [Subnet Participation](#subnet-participation)
  - [Makefile](#makefile)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [About the Rewards Mechanism](#about-the-rewards-mechanism)
- [Roadmap](#roadmap)
- [Compute Requirements](#compute-requirements)
- [License](#license)

---
## Introduction

CoinMetrics Blurb

---
## Design Decisions
Another Blurb

---
## Installation

First, install PM2:
```
sudo apt update
sudo apt install nodejs npm
sudo npm install pm2@latest -g
```
Verify installation:
```
pm2 --version
```


Clone the repository:
```
git clone https://github.com/coinmetrics/precog.git
cd precog
```

Create and source a python virtual environment:
```
python3 -m venv
source .venv/bin/activate
```

Install the requirements with poetry:
```
pip install poetry
poetry install
```

---
## Subnet Participation
heres how you do stuff

### Makefile
Start by editing the Makefile with you wallet and network information.

### Running a Miner
TODO: write this \
Base miner:
1. Run the command:
    ```
    make miner
    ```

Custom miner:
1. Write a custom forward function stored in precog/miners/your_function.py
    - This function should handle how the miner responds to requests from the validator
    - Within the function, synapse.predictions and synapse.interval should be set.
    - See [forward.py](https://github.com/coinmetrics/precog/blob/master/precog/miners/forward.py) for an example
2. Add a command to Makefile.
    - copy the miner command and rename it (e.g. miner_custom) in Makefile
    - replace the --forward_function argument with your_function
3. Run the Command:
    ```
    make miner_custom
    ```


### Running a Validator
```
make validator
```


## About the Rewards Mechanism
Optional but recommended.

---
## Roadmap

## Compute Requirements

TODO: update these
| Validator |   Miner   |
|---------- |-----------|
|  8gb RAM  |  8gb RAM  |
|  2 vCPUs  |  2 vCPUs  |

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 CoinMetrics LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
