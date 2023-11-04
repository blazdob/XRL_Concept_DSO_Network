import matplotlib.pyplot as plt
import pandapower as pp
from tqdm import tqdm
import pandas as pd
import time
from copy import deepcopy
import matplotlib.animation as animation
from dtaidistance import dtw
import datetime

#Using UpdateTrace figure
import numpy as np
from pandapower.plotting import cmap_continuous, cmap_discrete, create_line_collection, create_bus_collection, draw_collections
####################

fig, axs = plt.subplots(2, 2, figsize=(20, 10))
gridspec1 = axs[1][0].get_subplotspec()
gridspec2 = axs[0][0].get_subplotspec()

subfig1 = fig.add_subfigure(gridspec1)
subfig2 = fig.add_subfigure(gridspec2)

axsLeftBotom = subfig1.subplots(1, 3, sharey=True)
axsLeftTop = subfig2.subplots(2, 1, sharey=False)

gridspec3 = axsLeftTop[1].get_subplotspec()
# add a subfigure for the actions
subfig3 = subfig2.add_subfigure(gridspec3)
axsLeftTopActions = subfig3.subplots(1, 3, sharey=True)


active_consumers_ids = [1, 2, 5, 10, 11, 15, 16]
# net = pp.from_json('src/data/env_data/Srakovlje 2.json')
net = pp.from_json('src/data/env_data/srakovlje_DRIFT_geo1.json')
active_consumer_busses = list(net.load.loc[active_consumers_ids].bus.values)
active_consumer_busses = active_consumer_busses[::-1]
# print(active_consumer_busses)
# print(net.load)
# print(net.line_geodata)
loads = pd.read_csv('/Users/blazdobravec/Documents/FACULTY/DOC/RW1/PERSONAL/RL_Concept_DSO_Network/src/data/env_data/loads_and_generation_train.csv', index_col=0)
net.load.p_mw = 0
net.ext_grid.vm_pu = 1.03

base_net = deepcopy(net)
# Initialize communication with TMP102
path = "/Users/blazdobravec/Documents/FACULTY/DOC/RW1/PERSONAL/RL_Concept_DSO_Network/src/data/env_data/voltage_train_data.csv"
volgate_df = pd.read_csv(path, index_col=0)

path = "/Users/blazdobravec/Documents/FACULTY/DOC/RW1/PERSONAL/RL_Concept_DSO_Network/src/data/env_data/updated_voltage_train_data.csv"
updated_volgate_df = pd.read_csv(path, index_col=0)

max_unchangeds = volgate_df.max(axis=1)
min_unchangeds = volgate_df.min(axis=1)

###############
max_voltage = 1.05
path = "/Users/blazdobravec/Documents/FACULTY/DOC/RW1/PERSONAL/RL_Concept_DSO_Network/src/data/env_data/action_train_data.csv"
df_act = pd.read_csv(path, index_col=0)
df_act.columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
act_colors = ["r", "g", "b", "k", "y", "c", "tab:olive"]


###### CENTROIDS
# load numpy
centroids = np.load('/Users/blazdobravec/Documents/FACULTY/DOC/RW1/PERSONAL/RL_Concept_DSO_Network/src/data/env_data/centroids.npy', allow_pickle=True)
# centroids = centroids - 1.03000001
# average actions
avg_actions = np.load('/Users/blazdobravec/Documents/FACULTY/DOC/RW1/PERSONAL/RL_Concept_DSO_Network/src/data/env_data/action_centroids.npy', allow_pickle=True)

ax2 = axs[0][1].twinx()

xs = []

max_meas_V = []
new_max_meas_V = []
actions = []
max_V = []
min_meas_V = []

def animate(i, xs, max_meas_V, actions, new_max_meas_V, max_V, min_meas_V):

    #### ANIMATING NETWORK ####
    # 1. change the load
    date = loads.index[i]
    # minus 2 hours because of the time difference where we have day-month-year hh:mm format
    date = datetime.datetime.strptime(date, '%d.%m.%Y %H:%M') - datetime.timedelta(hours=5)
    
    action = df_act.iloc[i].values
    # load = loads.iloc[i].values * 9.5
    volgate = volgate_df.iloc[i].values
    updated_voltage = updated_volgate_df.iloc[i].values

    if i >= 48:
        voltage_net = volgate_df.iloc[i-48].values
        actions_net = df_act.iloc[i-48].values
        # 2. change the voltage
        base_net.res_bus.vm_pu = voltage_net

        # Draw x and y lists
        axsLeftTop[0].clear()
        cmap_list = [(0.95, "blue"), (1.02, "green"), (1.06, "red")]
        cmap, norm = cmap_continuous(cmap_list)
        bc = create_bus_collection(base_net, size=5, cmap=cmap, norm=norm)
        
        # Line colors
        cmap_list = [((0, 10), "green"), ((10, 30), "yellow"), ((30, 100), "red")]
        cmap, norm = cmap_discrete(cmap_list)
        lc = create_line_collection(base_net, cmap=cmap, norm=norm, use_bus_geodata=True)

        active_buses_collection = []
        for k, a in enumerate(actions_net):
            if a > 0:
                size = a*30
                bus = active_consumer_busses[k]
                col = act_colors[k]
                active_bus_col = create_bus_collection(base_net, buses=[bus], size=size, patch_type="rect", norm=norm, color=col)
                active_buses_collection.append(active_bus_col)
        if len(active_buses_collection) > 0:
            # draw_collections 
            collections = [bc, lc] + active_buses_collection
            draw_collections(collections, ax=axsLeftTop[0], draw=False, plot_colorbars=False)
            # draw_collections([bc, lc], ax=axsLeftTop[1], draw=False, plot_colorbars=False)
        else:
            draw_collections([bc, lc], ax=axsLeftTop[0], draw=False, plot_colorbars=False)
            # draw_collections([bc, lc], ax=axsLeftTop[1], draw=False, plot_colorbars=False)
        # axsLeftTop.set_title(date)
        axsLeftTop[0].set_title("Activations: " + str(date))
        # axsLeftTop[1].set_title("Voltage situation")
    else:
        axsLeftTop[0].clear()
        axsLeftTop[0].set_title("Activations: " + str(date))
        # axsLeftTop[1].set_title("Voltage situation")
    #### END OF ANIMATING NETWORK ####



    #### ANIMATING PLOTS ####
    # Add x and y to lists
    xs.append(i)
    max_meas_V.append(np.max(volgate))
    new_max_meas_V.append(np.max(updated_voltage))
    min_meas_V.append(np.min(volgate))
    actions.append(action.T)
    max_V.append(max_voltage)

    
    # Limit x and y lists to 20 items
    xs = xs[-96:]
    max_meas_V = max_meas_V[-96:]
    new_max_meas_V = new_max_meas_V[-96:]
    min_meas_V = min_meas_V[-96:]
    actions = actions[-96:]
    max_V = max_V[-96:]

    # 1. calculate the distance between the centroids and the max_meas_V
    # if len(max_meas_V) == 96:
    max_meas_V = np.array(max_meas_V)
    distances = []
    for centroid in centroids:
        # get only positive values of the centroid
        centroid = np.reshape(centroid, (1, 96))[0]

        # get first positive value
        first_positive_value_cent = np.where(centroid > 1.03)[0]
        if len(first_positive_value_cent) == 0:
            continue
        first_positive_value_cent = first_positive_value_cent[0]
        # get last positive value
        last_positive_value_cent = np.where(centroid > 1.03)[0]
        if len(last_positive_value_cent) == 0:
            continue
        last_positive_value_cent = last_positive_value_cent[-1]
        cropped_centroid = centroid[first_positive_value_cent:last_positive_value_cent].copy()

        first_positive_value_meas = np.where(max_meas_V > 1.03)[0]
        if len(first_positive_value_meas) == 0:
            continue
        first_positive_value_meas = first_positive_value_meas[0]
        # get last positive value
        last_positive_value_meas = np.where(max_meas_V > 1.03)[0]
        if len(last_positive_value_meas) == 0:
            continue
        last_positive_value_meas = last_positive_value_meas[-1]
        cropped_max_meas_V = max_meas_V[first_positive_value_meas:last_positive_value_meas].copy()
        
        cropped_max_meas_V -= 1.03
        cropped_centroid -= 1.03
        try:
            distance = dtw.distance_fast(cropped_centroid, cropped_max_meas_V, use_pruning=True)
            distances.append(distance)
        except:
            pass
    if len(distances) != 3:
        distances = [0, 0, 0]
        min_index = -1
    else:
        # find max index
        min_index = np.argmin(distances)

    # Draw x and y lists
    axs[0][1].clear()
    axs[0][1].plot(xs, max_meas_V, 'g-')
    axs[0][1].plot(xs, new_max_meas_V, 'r-')
    axs[0][1].plot(xs, max_V, linestyle='--', color='b')
    # draw vertical line in the middle
    axs[0][1].axvline(x=i-48, color='k', linestyle='--')
    # Format plot
    axs[0][1].set_xlabel('timestamps [15min]')
    axs[0][1].set_ylabel('Voltage [p.u.]')
    axs[0][1].set_title('Voltages')
    axs[0][1].legend(['max measured voltage', 'max adjusted voltage', 'voltage limit'], loc='upper left')
    # axs[0][1].plot(xs, min_meas_V, alpha=0.3, linestyle='--')
    axs[1][1].clear()
    axs[1][1].plot(xs, actions, alpha=0.5)
    # set colors of the actions
    for o in range(7):
        axs[1][1].lines[o].set_color(act_colors[o])
    axs[1][1].set_title('Actions')
    axs[1][1].set_ylabel('Actions percentage [%]')
    axs[1][1].set_xlabel('timestamps [15min]')
    axs[1][1].legend(['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'], loc='upper left')


    # plot the centroids
    #actions
    # calculate dtw distance between the action centroids in avg_actions and the actions in actions
    # action_distances = []
    # for action_centroid in avg_actions:
    #     action_centroid = np.reshape(action_centroid, (7, 96))[0]
    #     # get first positive value
    #     first_positive_value_cent = np.where(action_centroid > 0)[0]
    #     if len(first_positive_value_cent) == 0:
    #         continue
    #     first_positive_value_cent = first_positive_value_cent[0]
    #     # get last positive value
    #     last_positive_value_cent = np.where(action_centroid > 0)[0]
    #     if len(last_positive_value_cent) == 0:
    #         continue
    #     last_positive_value_cent = last_positive_value_cent[-1]
    #     cropped_centroid = action_centroid[first_positive_value_cent:last_positive_value_cent].copy()

    #     first_positive_value_meas = np.where(actions[-1] > 0)[0]
    #     if len(first_positive_value_meas) == 0:
    #         continue
    #     first_positive_value_meas = first_positive_value_meas[0]
    #     # get last positive value
    #     last_positive_value_meas = np.where(actions[-1] > 0)[0]
    #     if len(last_positive_value_meas) == 0:
    #         continue
    #     last_positive_value_meas = last_positive_value_meas[-1]
    #     cropped_max_meas_V = actions[-1][first_positive_value_meas:last_positive_value_meas].copy()
        
    #     try:
    #         action_distance = dtw.distance_fast(cropped_centroid, cropped_max_meas_V, use_pruning=True)
    #         action_distances.append(action_distance)
    #     except:
    #         pass
    # print(action_distances)

    for m, action_centroid in enumerate(avg_actions):
        axsLeftTopActions[m].clear()
        if m == min_index:
            axsLeftTopActions[m].plot(action_centroid, alpha=1)
            # running line
        else:
            axsLeftTopActions[m].plot(action_centroid, alpha=0.2)
        for n in range(7):
            axsLeftTopActions[m].lines[n].set_color(act_colors[n])
    #states
    for l, state_centroid in enumerate(centroids):
        axsLeftBotom[l].clear()
        if l == min_index:
            axsLeftBotom[l].plot(state_centroid, color=f"C{l}", alpha=1)
            axsLeftBotom[l].axhline(y=1.05, color='b', linestyle='--', alpha=1)
            axsLeftBotom[l].set_title(f"Scenario {l} with distance: {round(distances[l], 3)}", fontsize=11, color='green')
        else:
            axsLeftBotom[l].plot(state_centroid, color=f"C{l}", alpha=0.2)
            axsLeftBotom[l].axhline(y=1.05, color='b', linestyle='--', alpha=0.2)
            axsLeftBotom[l].set_title(f"Scenario {l} with distance: {round(distances[l], 3)}", fontsize=9, color='red')
    # for l, ax in enumerate(axsLeftBotom):
    #     # plot axhline
    #     #plot running line .axvline(x=i-48, color='k', linestyle='--')
    #     if l == min_index:
    #         ax.plot(centroids[l], color=f"C{l}", alpha=1)
    #         ax.axhline(y=1.05, color='b', linestyle='--', alpha=1)
    #         ax.set_title(f"Scenario {l} with distance: {round(distances[l], 3)}", fontsize=11, color='green')
    #     else:
    #         ax.plot(centroids[l], color=f"C{l}", alpha=0.2)
    #         ax.axhline(y=1.05, color='b', linestyle='--', alpha=0.2)
    #         ax.set_title(f"Scenario {l} with distance: {round(distances[l], 3)}", fontsize=9, color='red')
        axsLeftBotom[l].set_xlabel('timestamps [15min]')
        axsLeftBotom[l].set_ylabel('Voltage [p.u.]')

    # print that you are itterating the next 100 frames
    if i%100 == 0:
        print(f"itterating {i} frames")

    

ani = animation.FuncAnimation(fig, animate, fargs=(xs, max_meas_V, actions, new_max_meas_V, max_V, min_meas_V), interval=1, frames=3000)
# plt.show()
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
f = r"src/results_data/animation_{}.avi".format(date)
writervideo = animation.FFMpegWriter(fps=2)
ani.save(f, writer=writervideo)
