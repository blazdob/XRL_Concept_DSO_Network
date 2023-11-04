import matplotlib.pyplot as plt
import pandapower as pp
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.animation as animation

#Using UpdateTrace figure
import numpy as np
from pandapower.plotting import cmap_continuous, cmap_discrete, create_line_collection, create_bus_collection, draw_collections
####################

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

active_consumers_ids = [1, 2, 5, 10, 11, 15, 16]
# net = pp.from_json('src/data/env_data/Srakovlje 2.json')
net = pp.from_json('src/data/env_data/srakovlje_DRIFT_geo1.json')
active_consumer_busses = list(net.load.loc[active_consumers_ids].bus.values)
# print(net.line_geodata)
load = pd.read_csv('src/data/env_data/loads_and_generation_test.csv', index_col=0)
net.load.p_mw = 0
net.ext_grid.vm_pu = 1.03

# Initialize communication with TMP102
path = "/Users/blazdobravec/Documents/WORK/INTERNI-PROJEKTI/DRIFT/DEVELOPMENT/PPO-for-Beginners/environment/data/srakovlje/voltage_test_data.csv"
volgate_df = pd.read_csv(path, index_col=0)

max_unchangeds = volgate_df.max(axis=1)
min_unchangeds = volgate_df.min(axis=1)

###############
max_voltage = 1.05
path = "/Users/blazdobravec/Documents/WORK/INTERNI-PROJEKTI/DRIFT/DEVELOPMENT/PPO-for-Beginners/environment/data/srakovlje/action_test_data.csv"
df_act = pd.read_csv(path, index_col=0)
df_act.columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]

ax2 = axs[1].twinx()

xs = []

max_meas_V = []
actions = []
max_V = []
min_meas_V = []

def animate(i, xs, max_meas_V, actions, max_V, min_meas_V):

    #### ANIMATING NETWORK ####
    # 1. change the load
    # net.load.p_mw = load.iloc[i].values * 9.5
    date = load.index[i]
    # pp.runpp(net)
    # 2. change the voltage
    net.res_bus.vm_pu = volgate_df.iloc[i].values

    # Draw x and y lists
    axs[0].clear()
    cmap_list = [(0.93, "blue"), (1.0, "green"), (1.07, "red")]
    cmap, norm = cmap_continuous(cmap_list)
    bc = create_bus_collection(net, size=5, cmap=cmap, norm=norm)
    # Line colors
    cmap_list = [((0, 10), "green"), ((10, 30), "yellow"), ((30, 100), "red")]
    cmap, norm = cmap_discrete(cmap_list)
    lc = create_line_collection(net, cmap=cmap, norm=norm, use_bus_geodata=True)
    # draw_collections 
    draw_collections([bc, lc], ax=axs[0], draw=False, plot_colorbars=False)
    axs[0].set_title(date)

    #### END OF ANIMATING NETWORK ####

    #### ANIMATING PLOT ####
    # Add x and y to lists
    xs.append(i)
    max_meas_V.append(volgate_df.iloc[i].max())
    min_meas_V.append(volgate_df.iloc[i].min())
    actions.append(list(df_act.iloc[i].values))
    max_V.append(max_voltage)

    # Limit x and y lists to 20 items
    xs = xs[-96:]
    max_meas_V = max_meas_V[-96:]
    min_meas_V = min_meas_V[-96:]
    actions = actions[-96:]
    max_V = max_V[-96:]

    # Draw x and y lists
    axs[1].clear()
    ax2.clear()
    axs[1].plot(xs, max_meas_V, 'g-')
    axs[1].plot(xs, max_V, linestyle='--')
    axs[1].plot(xs, min_meas_V, alpha=0.1, linestyle='--')
    ax2.plot(xs, actions, alpha=0.5)
    # Format plot
    axs[1].set_xlabel('timestamps [15min]')
    axs[1].set_ylabel('Voltage [V]')
    ax2.set_ylabel('Actions percentage [%]')

    



# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, max_meas_V, actions, max_V, min_meas_V), interval=100)
plt.show()
