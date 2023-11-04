import numpy as np
import matplotlib.pyplot as plt


def bowl(vs, v_ref=1.0, scale=.1):
    def normal(v, loc, scale):
        return 1 / np.sqrt(2 * np.pi * scale**2) * np.exp( - 0.5 * np.square(v - loc) / scale**2 )
    def _bowl(v):
        if np.abs(v-v_ref) > 0.05:
            return 2 * np.abs(v-v_ref) - 0.095
        else:
            return - 0.01 * normal(v, v_ref, scale) + 0.04
    return np.array([_bowl(v)*10 for v in vs])



def bump(vs):
    def _bump(v):
        if np.abs(v) < 1:
            return np.exp( - 1 / (1 - v**4) )
        elif 1 < v < 3:
            return np.exp( - 1 / (1 - ( v - 2 )**4 ) )
        else:
            return 0.0
    return np.array([_bump(v)*19 for v in vs])


def courant_beltrami(vs, v_lower=0.95, v_upper=1.05):
    def _courant_beltrami(v):
        return np.square(max(0, v - v_upper)) + np.square(max(0, v_lower - v))
    return np.array([_courant_beltrami(v)*10 for v in vs])

def l1(vs, v_ref=1.0):
    def _l1(v):
        return np.abs( v - v_ref )
    return np.array([_l1(v)*10 for v in vs])

def l2(vs, v_ref=1.0):
    def _l2(v):
        return (2 * np.square(v - v_ref))
    return np.array([_l2(v)*50 for v in vs])

Voltage_barrier = dict(
    l1=l1,
    l2=l2,
    bowl=bowl,
    bump=bump,
    courant_beltrami=courant_beltrami
)

if __name__ == "__main__":
    #plot_voltage_barrier(Voltage_Barrier["l1"], 0.5, 0.5)
    # plot voltage barier
    vs = np.linspace(0.9, 1.1, 30)
    #random shuffle
    # np.random.shuffle(vs)
    y = l1(vs)
    print(y)
    plt.plot(vs, y)
    plt.show()
