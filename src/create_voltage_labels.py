import pandas as pd
import pandapower as pp
from tqdm import tqdm
from scipy import optimize
from scipy.optimize import Bounds
from copy import deepcopy

train = pd.read_csv('src/data/env_data/loads_and_generation_train.csv', index_col=0, parse_dates=True)
test = pd.read_csv('src/data/env_data/loads_and_generation_test.csv', index_col=0, parse_dates=True)
base_net = pp.from_json('src/data/env_data/srakovlje_DRIFT_geo1.json')


# test.columns = test.columns.astype(int)
test.reset_index(inplace=True, drop=True)
train.reset_index(inplace=True, drop=True)

# get busses of the train.columns
busses = base_net.load[["name", "bus"]]
gis_ids = test.columns.values

# create mapping of gis_id to bus
mapping = {}
for name in tqdm(base_net.load.name):
    if name.split(" ")[1] in gis_ids:
        mapping[name.split(" ")[1]] = (base_net.load[base_net.load.name == name].bus.values[0], base_net.load[base_net.load.name == name].index.values[0])
    else:
        print(name)

# rename columns to load indices
test.columns = [mapping[gis_id][1] for gis_id in test.columns]
train.columns = [mapping[gis_id][1] for gis_id in train.columns]

def minimisation(action, args=None):
    # locate the busses that are more than 1.05
    base_net = args
    net = deepcopy(base_net)
    actives = [1, 2, 5, 10, 11, 15, 16]
    #locate busses where voltage is bigger than 1.05
    critical = net.res_bus[net.res_bus.vm_pu > 1.05].index.values

    updated_loads = deepcopy(net.load.p_mw)
    for i, active_i in enumerate(actives):
        if updated_loads[active_i] <= 0.0:
            updated_loads[active_i] = updated_loads[active_i] * (1 - action[i])

    net.load.p_mw = updated_loads
    pp.runpp(net)

    # calculate loss function
    loss = 0
    for critical_bus in critical:
        loss += (net.res_bus.vm_pu[critical_bus] - 1.05) ** 2
    net = deepcopy(base_net)
    return loss

# drop column if the value is more than 1000
def run_optimisation(base_net, data):
    steps = len(data)
    actives = [1, 2, 5, 10, 11, 15, 16]
    voltage_data = {}
    updated_voltage_data = {}
    action_data = {}

    bounds = Bounds([0,0,0,0,0,0,0], [1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    net = deepcopy(base_net)
    print(net.ext_grid)
    for step in tqdm(range(steps)):
        step_consumption = data.iloc[step].values
        # increase the negative values
        # step_consumption[step_consumption < 0] = step_consumption[step_consumption < 0]
        net.load.p_mw = step_consumption
        try:
            pp.runpp(net)
            voltage_data[step] = net.res_bus.vm_pu
            # locate only busses where loads are connected
            if net.res_bus.vm_pu.max().max() > 1.05:
                print(net.res_bus.vm_pu.max().max())
                minimal = optimize.minimize(minimisation, x0=[0.001,0.001,0.001,0.001,0.001,0.001,0.001], args=net, bounds=bounds)
                action = minimal.x
                # apply the action to the network
                base_net.load.p_mw = step_consumption

                updated_loads = deepcopy(base_net.load.p_mw)

                for i, active_i in enumerate(actives):
                    if updated_loads[active_i] <= 0.0:
                        updated_loads[active_i] = updated_loads[active_i] * (1 - action[i])
                base_net.load.p_mw = updated_loads

                pp.runpp(base_net)
                updated_voltage_data[step] = base_net.res_bus.vm_pu

            else:
                action = [0,0,0,0,0,0,0]
                updated_voltage_data[step] = net.res_bus.vm_pu
            action_data[step] = action
        except pp.LoadflowNotConverged:
            print("Error in step {}".format(step))
            action = [0,0,0,0,0,0,0]
            updated_voltage_data[step] = net.res_bus.vm_pu
            voltage_data[step] = net.res_bus.vm_pu
            action_data[step] = action
    return voltage_data, updated_voltage_data, action_data




print("Test data.....")
voltage_test_data, updated_test_voltage_data, action_test_data = run_optimisation(base_net, test)
# # save the test data
voltage_test_data = pd.DataFrame(voltage_test_data).T
voltage_test_data.to_csv('src/data/env_data/voltage_test_data.csv')

updated_test_voltage_data = pd.DataFrame(updated_test_voltage_data).T
updated_test_voltage_data.to_csv('src/data/env_data/updated_test_voltage_data.csv')

action_test_data = pd.DataFrame(action_test_data).T
action_test_data.to_csv('src/data/env_data/action_test_data.csv')

print("Train data.....")
voltage_train_data, updated_voltage_train_data, action_train_data = run_optimisation(base_net, train)
# save the train data
voltage_train_data = pd.DataFrame(voltage_train_data).T
voltage_train_data.to_csv('src/data/env_data/voltage_train_data.csv')

updated_voltage_train_data = pd.DataFrame(updated_voltage_train_data).T
updated_voltage_train_data.to_csv('src/data/env_data/updated_voltage_train_data.csv')

action_train_data = pd.DataFrame(action_train_data).T
action_train_data.to_csv('src/data/env_data/action_train_data.csv')

