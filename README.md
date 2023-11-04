# XRL Concept DSO network
The repository contains the code for the reinforcement learning concept for the DSO network voltage control.

## Installation
The code is written in Python 3.8+ and uses the following packages:
- numpy
- scipy
- matplotlib
- pandas
- pyyaml
- pandapower

The code is tested on Windows 10 and macOS Ventura 13.1.

### Windows
1. Install Python 3.8+ from https://www.python.org/downloads/
2. Create a virtual environment for the project
```
python -m venv venv_drift
```
3. Activate the virtual environment
```
venv_drift\Scripts\activate.bat
```
4. Install the packages listed in the requirements.txt file
```
pip install -r requirements.txt
```

### macOS
1. Install Python 3.8+ from https://www.python.org/downloads/
2. Create a virtual environment for the project
```
python3 -m venv venv_drift
```
3. Activate the virtual environment
```
source venv_drift/bin/activate
```
4. Install the packages listed in the requirements.txt file
```
pip install -r requirements.txt
```

## Usage
The code is structured in the following way:
- `src` contains the source code
    - `data` contains the input data
    - `agents` contains the agent classes
    - `env` contains the environment class
    - `main.py` is the main file to run the simulation
    - `results` contains the results of the simulation

The simulation can be run by executing the `main.py` file. The simulation parameters can be set in the `config.yaml` file.
Different scenarios can be run by changing the `scenario` parameter in the `config.yaml` file.

## Running the tests

ROOT_DIR = base directory of the project (not src/)

Prepare the data and the environment:
```
python src/data/data_prep.py
```
The function creates the network and constructs the database of the agents behaviour for learning.
The results are then saved in the `data/env_data` folder.

Run the tests:
```
python src/main.py --train True 
                    --network_path src/data/env_data/Srakovlje.json 
                    --load_train_file src/data/env_data/loads_and_generation.csv
                    --load_test_file src/data/env_data/loads_and_generation_test.csv
                    --voltage_barier_type l1 
                    --voltage_threshold 0.2 
                    --episode_limit = 288
                    --num_episodes = 400
                    --tensorboard_log True 
                    --tensorboard_log_path src/voltage_control_tensorboard/ 
                    --model_path src/saved_models/voltage_control
```

If you want to only run the already learned model, set the `--train` parameter to `False` and set the `--model_path` parameter to the path of the saved model. A prerequesite for this is that the `--model_path` and `--load_test_file` parameters are set to the values that the model objects and test timeseries data.

## License
[MIT]

Copyright (c) [2022] [EG, FE-UL]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Authors
- [Klemen Knez]()
- [Blaž Dobravec]()
- [Jure Žabkar]()
- [Boštjan Blažič]()
- [Vitomir Štruc]()








