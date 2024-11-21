<div align="center">
<img src="docs/images/precog-logo.png" />

# **CoinMetrics Precog Subnet** <!-- omit in toc -->

|     |     |
| :-: | :-: |
Badges here?


</div>

---

- [Introduction](#introduction)
- [Design Decisions](#design-decisions)
- [Installation](#installation)
  - [Install PM2](#install-pm2)
  - [Install-Repo](#install-repo)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
  - [Running Miner/Validator in Docker](#running-minervalidator-in-docker)
- [About the Rewards Mechanism](#about-the-rewards-mechanism)
- [Compute Requirements](#compute-requirements)
- [Roadmap](#roadmap)
- [License](#license)

---
## Introduction

CoinMetrics Blurb

---
## Design Decisions
Another Blurb

---
## Installation

### Install PM2
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


### Install-Repo

Begin by cloning the repository:
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

### Running a Miner
TODO: write this

### Running a Validator
TODO: write this


## About the Rewards Mechanism
Optional but recommended.

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
# Copyright © 2024 Foundry Digital LLC

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
