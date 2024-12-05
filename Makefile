## Network Parameters ##
finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
locanet = ws://127.0.0.1:9944

testnet_netuid = 256
localnet_netuid = 1
logging_level = trace # options= ['info', 'debug', 'trace']

netuid = $(testnet_netuid)
network = $(testnet)

## User Parameters
coldkey = default
validator_hotkey = validator
miner_hotkey = miner

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
		--wallet.hotkey $(validator_hotkey) \
		--network $(network) \
		--axon.port 30335 \
		--netuid $(netuid) \
		--logging.level $(logging_level)

miner:
	python start_miner.py \
		--neuron.name miner \
		--wallet.name $(coldkey) \
		--wallet.hotkey $(miner_hotkey) \
		--network $(network) \
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
		--network $(network) \
		--axon.port 30337 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--timeout 16 \
		--forward_function forward_bad

