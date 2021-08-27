CONFIG_FILE=./settings/config_files/local_config.ini

run-local:
	python3 main.py --config_file ${CONFIG_FILE} 

run-testing:
	python3 main_testing.py --config_file ${CONFIG_FILE} 

run-requirements:
	pip3 install -r requirements.txt

quick-start: 
	run-requirements run-local
