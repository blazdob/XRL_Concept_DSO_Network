import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

xs = []

max_meas_V = []
actions = []
max_V = []
min_meas_V = []

max_voltage = 1.05

# Initialize communication with TMP102
path = "/Users/blazdobravec/Documents/WORK/INTERNI-PROJEKTI/DRIFT/DEVELOPMENT/PPO-for-Beginners/environment/data/srakovlje/voltage_train_data.csv"
volgate_df = pd.read_csv(path, index_col=0)

max_unchangeds = volgate_df.max(axis=1)
min_unchangeds = volgate_df.min(axis=1)

###############
path = "/Users/blazdobravec/Documents/WORK/INTERNI-PROJEKTI/DRIFT/DEVELOPMENT/PPO-for-Beginners/environment/data/srakovlje/action_train_data.csv"
df = pd.read_csv(path, index_col=0)
df.columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]

# This function is called periodically from FuncAnimation
def animate(i, xs, max_meas_V, min_meas_V, actions, max_V):

    # Add x and y to lists
    xs.append(i)
    max_meas_V.append(volgate_df.iloc[i].max())
    min_meas_V.append(volgate_df.iloc[i].min())
    actions.append(list(df.iloc[i].values))
    max_V.append(max_voltage)

    # Limit x and y lists to 20 items
    xs = xs[-96:]
    max_meas_V = max_meas_V[-96:]
    min_meas_V = min_meas_V[-96:]
    actions = actions[-96:]
    max_V = max_V[-96:]

    # Draw x and y lists
    ax1.clear()
    ax2.clear()
    ax1.plot(xs, max_meas_V, 'g-')
    ax1.plot(xs, max_V, 'b-', linestyle='--')
    ax1.plot(xs, min_meas_V, alpha=0.1, linestyle='--')
    ax2.plot(xs, actions, alpha=0.5)
    # Format plot
    ax1.set_xlabel('timestamps [15min]')
    ax1.set_ylabel('Voltage [V]')
    ax2.set_ylabel('Actions percentage [%]')
    # plt.xticks(rotation=45, ha='right')
    # plt.subplots_adjust(bottom=0.30)


# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, max_meas_V, min_meas_V, actions, max_V), interval=500)
plt.show()