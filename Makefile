network = ws://127.0.0.1:9944
netuid = 1
logging_level = trace # options= ['info', 'debug', 'trace']
coldkey = cm-owner

metagraph:
	btcli subnet metagraph --netuid $(netuid) --subtensor.chain_endpoint $(network)

register:
	{ \
		read -p 'Wallet name?: ' wallet_name ;\
		read -p 'Hotkey?: ' hotkey_name ;\
		btcli subnet register --netuid $(netuid) --wallet.name "$$wallet_name" --wallet.hotkey "$$hotkey_name" --subtensor.chain_endpoint $(network) ;\
	}
validator:
	python start_validator.py \
		--neuron.name validator \
		--wallet.name $(coldkey) \
		--wallet.hotkey validator \
		--subtensor.chain_endpoint $(network) \
		--axon.port 30335 \
		--netuid $(netuid) \
		--logging.level $(logging_level)

validator2:
	python start_validator.py \
		--neuron.name validator2 \
		--wallet.name $(coldkey) \
		--wallet.hotkey validator2 \
		--subtensor.chain_endpoint $(network) \
		--axon.port 30339 \
		--netuid $(netuid) \
		--logging.level $(logging_level)

miner:
	python start_miner.py \
		--neuron.name miner \
		--wallet.name $(coldkey) \
		--wallet.hotkey miner \
		--subtensor.chain_endpoint $(network) \
		--axon.port 30336 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--timeout 16 \
		--forward_function forward

miner2:
	python start_miner.py \
		--neuron.name miner2 \
		--wallet.name $(coldkey) \
		--wallet.hotkey miner2 \
		--subtensor.chain_endpoint $(network) \
		--axon.port 30337 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--timeout 16 \
		--forward_function forward_bad

miner3:
	python start_miner.py \
		--neuron.name miner3 \
		--wallet.name $(coldkey) \
		--wallet.hotkey miner3 \
		--subtensor.chain_endpoint $(network) \
		--axon.port 30338 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--timeout 16 \
		--forward_function forward

setup_local:
	btcli wallet faucet --wallet.name $(coldkey) --subtensor.chain_endpoint $(network) ;\
	btcli subnet create --wallet.name $(coldkey) --subtensor.chain_endpoint $(network) ;\
	btcli subnet register \
		--wallet.name $(coldkey) \
		--wallet.hotkey validator \
		--netuid $(netuid)
		--subtensor.chain_endpoint $(network) ;\
	btcli stake add --wallet.name $(coldkey) --wallet.hotkey validator --amount 1024 ;\
