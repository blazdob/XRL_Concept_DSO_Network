from .voltage_barrier_functions import Voltage_barrier



class VoltageBarrier(object):
    def __init__(self, name):
        self.name = name
        self.voltage_barrier = Voltage_barrier[name]

    def step(self, vs):
        return 1-self.voltage_barrier(vs)

    def min(self):
        return 1-self.voltage_barrier([1.1])
    
    def max(self):
        return 1-self.voltage_barrier([1])


if __name__ == "__main__":
    #plot_voltage_barrier(Voltage_Barrier["l1"], 0.5, 0.5)
    # plot voltage barier
    import numpy as np
    import matplotlib.pyplot as plt
    vs = np.linspace(0.9, 1.1, 30)
    #random shuffle
    # np.random.shuffle(vs)
    vBarrier = VoltageBarrier("l1")
    y = vBarrier.step(vs)
    mean = np.mean(y)
    # print(y, mean)
    print(vBarrier.min(), vBarrier.max())
    plt.plot(vs, y)
    plt.show()