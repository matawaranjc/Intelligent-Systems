import numpy as np
import matplotlib.pyplot as plt

# Function for Izhikevich models of neurons
# Values from Izhikevich
# Regular spiking neuron and fast spiking neuron
# Parameters
# a - time scale of the recovery variable (u)
# b - sensitivity of the recovery variable (u)
# c - after-spike reset value of the membrane potential (v)
# d - after-spike reset of the recovery variable (u)
# model - regular spiking (RS) neuron and fast spiking (FS) neuron

# parameters value declaration for neural models
# first index of the list is for RS
# second index of the list is for FS

a = [0.02, 0.1]
b = [0.2, 0.2]
c = [-65, -65]
d = [8, 2]
trials = np.arange(41)
N_models = ["Regular Spiking Neuron", "Fast Spiking Neuron"]
MSR = np.zeros((2, 41))

def Izhikevich(a, b, c, d, model):

    steps = 1000  # this simulation runs for 1000 steps
    v = -64  # membrane potential
    VV = []
    u = b * v  # recovery variable
    uu = []
    VVplot = np.zeros((5, 4000))
    tau = 0.25  # tau is the discretization time-step
    I_series = [1, 10, 20, 30, 40]
    T1 = 0 # constant input from the beginning of each trial
    spike_ts = []
    meanSpikeRate = []
    tspan = np.arange(0, steps, tau)  # tspan is the simulation interval
    p = 0

    for i in range(0, 41):  # trials of synaptic currents inputs (I)
        for t in tspan:
            if t > T1:
                I = i
            else:
                I = 0
            v = v + tau*(0.04 * v ** 2 + 5 * v + 140 - u + I)  # calculates membrane potential
            u = u + tau * a * (b * v - u)  # calculates recovery variable

            if v >= 30:  # if this is a spike
                VV.append(30) # VV is the time-series of membrane potentials
                v = c
                u = u + d
                spike_ts.append(1) # records a spike
            else:
                VV.append(v)
                spike_ts.append(0) # records no spike
            uu.append(u)

        x = 0
        for s in spike_ts[200:]:  # to disregard the first 200 steps - Mean Spike rate Calculation
            if s == 1:
                x = x + 1
        R = x / 800
        meanSpikeRate.append(R)

        if i in I_series:
            VVplot[p] = VV # plotted as the output
            p = p + 1
        VV.clear()
        spike_ts.clear()


    # to build the graph for full 1,000-step time-series of the membrane potential v for each model

    fig, axes = plt.subplots(5)
    for i in range(5):
        axes[i].plot(tspan, VVplot[i], linewidth=0.5, color='green')
        axes[2].set_ylabel("Membrane potential (v)")
        axes[0].set_ylabel("I = 1")
        axes[1].set_ylabel("I = 10")
        axes[2].set_ylabel("Potential Membrane (v) \nI = 20")
        axes[3].set_ylabel("I = 30")
        axes[4].set_ylabel("I = 40")
        axes[i].grid(True)
        axes[i].set_ylim(-95, 50)
    fig.tight_layout()
    plt.suptitle(model, y=1, fontweight="bold")
    plt.xlabel("Time-step")
    plt.savefig(model + ".jpeg", dpi=300)
    plt.show()
    return meanSpikeRate



for opt in range(2):
    MSR[opt] = Izhikevich(a[opt], b[opt], c[opt], d[opt], N_models[opt])

# to build the graph for R vs I for RS

plt.title("R vs I for Regular Spiking", fontweight="bold")
plt.xlabel("Mean Spike Rate (MSR)")
plt.ylabel("Synaptic currents inputs (I)")
plt.plot(MSR[0], trials, label="Regular Spiking Neuron (RS)", color='green')
plt.grid(True)
plt.legend(loc="upper left")
plt.savefig('MSR_RS', dpi=300)
plt.show()

# to build the graph for R vs I for RS and FS

plt.title("R vs I (RS and FS)", fontweight="bold")
plt.plot(MSR[0], trials, label="Regular Spiking Neuron (RS)", color='green')
plt.plot(MSR[1], trials, label="Fast Spiking Neuron (FS)", color ='blue')
plt.xlabel("Mean Spike Rate (MSR)")
plt.ylabel("Synaptic currents inputs (I)")
plt.legend(loc="lower center")
plt.grid(True)
plt.savefig('MSR_RS_FS.jpeg', dpi=300)
plt.show()


def burstingNeuron(a, b, c, d):     # function to implement 2-neuron of chattering
    steps = 1000
    V = -65
    u = b*V
    VV1 = []
    VV1plot = np.zeros((5,4000))
    VV2 = []
    VV2plot = np.zeros((5,4000))
    n=0
    w_series = [0, 10, 20, 30, 40]
    uu = []
    tau = 0.25
    T1 = 0
    spike_ts =[]
    BMSR =[]
    Ia = 2          # steady input for neuron A
    Ib = 5          # steady input for neuron B
    tspan = np.arange(0,steps,tau)
    weight = [0, 10, 20, 30, 40]
    for w in weight:
        for t in tspan:
            if t>T1:
                I = w
            else:
                I = 0
            V = V + tau*(0.04*V**2+5*V+140-u+I)
            u = u + tau * a * (b * V - u)
            if V>30:
                Yb = 1
                Ya = 1
                Ia_total = Ia + -(w * Yb)
                Ib_total = Ib + (w * Ya)
                VV1.append(Ia_total)
                VV2.append(Ib_total)
                V = c
                u = u+d
                spike_ts.append(1)
            else:
                VV1.append(2)
                VV2.append(5)
                spike_ts.append(0)
            uu.append(u)
        cn=0
        for value in spike_ts[200:]:
            if value == 1:
                cn = cn+1
        R = cn/800
        if w in w_series:
            VV1plot[n] = VV1
            VV2plot[n] = VV2
            n = n + 1
        VV1.clear()
        VV2.clear()
    fig, axes = plt.subplots(5)
    for p in range(5):
        axes[p].plot(tspan, VV1plot[p], linewidth=0.5, label="Neuron A", color = 'blue')
        axes[p].plot(tspan, VV2plot[p], linewidth=0.5, label="Neuron B", color = 'green')
        plt.legend(loc='best')
        axes[0].set_ylabel("W = 0")
        axes[1].set_ylabel("W = 10")
        axes[2].set_ylabel("Potential Membrane (v) \nW = 20")
        axes[3].set_ylabel("W = 30")
        axes[4].set_ylabel("W = 40")
        axes[p].grid(True)
        axes[p].set_ylim(-95, 50)
    fig.tight_layout()
    plt.suptitle("2-neuron network of chattering", y=1, fontweight="bold")
    plt.xlabel("Time-step")
    plt.savefig("2-neuron.jpeg", dpi=300)
    plt.show()
    plt.show()
    return BMSR

CMSR = burstingNeuron(a=0.02, b=0.2, c=-50, d=2)