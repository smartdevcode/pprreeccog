################################################################################
#                               User Parameters                                #
################################################################################
coldkey = validator
# coldkey = miner
validator_hotkey = default
miner_hotkey = default
netuid = $(localnet_netuid)
network = $(localnet)
logging_level = trace


################################################################################
#                             Network Parameters                               #
################################################################################
finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
localnet = ws://127.0.0.1:9945

testnet_netuid = 256
localnet_netuid = 1
logging_level = debug # options= ['info', 'debug', 'trace']


################################################################################
#                                 Commands                                     #
################################################################################

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
		--subtensor.chain_endpoint $(network) \
		--axon.port 8091 \
		--netuid $(netuid) \
		--logging.level $(logging_level)

miner:
	python start_miner.py \
		--neuron.name miner \
		--wallet.name $(coldkey) \
		--wallet.hotkey $(miner_hotkey) \
		--subtensor.chain_endpoint $(network) \
		--axon.port 8092 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--timeout 16 \
		--vpermit_tao_limit 2 \
		--forward_function base_miner

custom_miner:
	python start_miner.py \
		--neuron.name custom_miner \
		--wallet.name $(coldkey) \
		--wallet.hotkey miner2 \
		--subtensor.chain_endpoint $(network) \
		--axon.port 8093 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--timeout 16 \
		--forward_function custom_function
