import networkx as nx
import numpy as np
import random

def simple_network_model(P, T, dt):
    del_range = P["del"].copy()
    eps_range = P["eps"].copy()

    X, Y = np.meshgrid(eps_range, del_range)

    t = np.arange(0.0, T + dt, dt, dtype=float)
    num_steps = len(t)

    # Initializing arrays to store my matrices
    S = np.zeros(shape=(num_steps, P["S"].shape[0], P["S"].shape[1]))
    I = np.zeros(shape=(num_steps, P["I"].shape[0], P["I"].shape[1]))
    R = np.zeros(shape=(num_steps, P["R"].shape[0], P["R"].shape[1]))

    # Initializing arrays to store my sums
    S_sum = np.zeros(shape=(num_steps))
    I_sum = np.zeros(shape=(num_steps))
    R_sum = np.zeros(shape=(num_steps))

    # Copying initial frequencies
    S[0] = P["S"].copy()
    I[0] = P["I"].copy()
    R[0] = P["R"].copy()

    S_sum[0] = np.sum(S[0] * P["G"])
    I_sum[0] = np.sum(I[0] * P["G"])
    R_sum[0] = np.sum(R[0] * P["G"])

    incident_inf = np.empty(shape=(num_steps - 1))

    for i in range(num_steps - 1):
        del_1 = np.sum(Y * I[i] * P["G"])

        S[i + 1] = S[i] + dt * (-P["beta"] * X * S[i] * P["k"] * del_1)
        I[i + 1] = I[i] + dt * (
            -P["gamma"] * I[i] + P["beta"] * X * S[i] * P["k"] * del_1
        )
        R[i + 1] = R[i] + dt * (P["gamma"] * I[i])

        S_sum[i + 1] = np.sum(S[i + 1] * P["G"])
        I_sum[i + 1] = np.sum(I[i + 1] * P["G"])
        R_sum[i + 1] = np.sum(R[i + 1] * P["G"])

        incident_inf[i] = np.sum(P["G"] * (S[i + 1] - S[i]) / dt)

    return [S_sum, I_sum, R_sum, incident_inf]


def network_model_v2(P, T, dt):

    del_range = np.linspace(0.1, 3, 100)  # Include in P later
    eps_range = np.linspace(0.1, 3, 100)  # Include in P later

    X, Y = np.meshgrid(eps_range, del_range)

    del_ls = []

    t = np.arange(0.0, T + dt, dt, dtype=float)
    num_steps = len(t)

    # Initializing arrays to store my matrices
    S1 = np.zeros(shape=(num_steps, P["S1"].shape[0], P["S1"].shape[1]))
    I1 = np.zeros(shape=(num_steps, P["I1"].shape[0], P["I1"].shape[1]))
    R1 = np.zeros(shape=(num_steps, P["R1"].shape[0], P["R1"].shape[1]))
    S2 = np.zeros(shape=(num_steps, P["S2"].shape[0], P["S2"].shape[1]))
    I2 = np.zeros(shape=(num_steps, P["I2"].shape[0], P["I2"].shape[1]))
    R2 = np.zeros(shape=(num_steps, P["R2"].shape[0], P["R2"].shape[1]))

    # Initializing arrays to store my sums
    S1_sum = np.zeros(shape=(num_steps))
    I1_sum = np.zeros(shape=(num_steps))
    R1_sum = np.zeros(shape=(num_steps))
    S2_sum = np.zeros(shape=(num_steps))
    I2_sum = np.zeros(shape=(num_steps))
    R2_sum = np.zeros(shape=(num_steps))

    # Copying initial frequencies
    S1[0] = P["S1"].copy()
    I1[0] = P["I1"].copy()
    R1[0] = P["R1"].copy()
    S2[0] = P["S2"].copy()
    I2[0] = P["I2"].copy()
    R2[0] = P["R2"].copy()

    S1_sum[0] = np.sum(S1[0] * P["g1"])
    I1_sum[0] = np.sum(I1[0] * P["g1"])
    R1_sum[0] = np.sum(R1[0] * P["g1"])
    S2_sum[0] = np.sum(S2[0] * P["g2"])
    I2_sum[0] = np.sum(I2[0] * P["g2"])
    R2_sum[0] = np.sum(R2[0] * P["g2"])

    incident_inf = np.empty(shape=(num_steps - 1, 2))

    for i in range(num_steps - 1):
        del_1 = np.sum(Y * I1[i] * P["g1"])
        del_2 = np.sum(Y * I2[i] * P["g2"])

        del_ls.append([del_1, del_2])

        S1[i + 1] = S1[i] + dt * (
            -P["beta"]
            * X
            * S1[i]
            * (
                ((P["k"] * (1 + P["e"])) / 2) * del_1
                + ((P["k"] * (1 - P["e"])) / 2) * del_2
            )
        )
        I1[i + 1] = I1[i] + dt * (
            -P["gamma"] * I1[i]
            + P["beta"]
            * X
            * S1[i]
            * (
                ((P["k"] * (1 + P["e"])) / 2) * del_1
                + ((P["k"] * (1 - P["e"])) / 2) * del_2
            )
        )
        R1[i + 1] = R1[i] + dt * (P["gamma"] * I1[i])

        S2[i + 1] = S2[i] + dt * (
            -P["beta"]
            * X
            * S2[i]
            * (
                ((P["k"] * (1 + P["e"])) / 2) * del_2
                + ((P["k"] * (1 - P["e"])) / 2) * del_1
            )
        )
        I2[i + 1] = I2[i] + dt * (
            -P["gamma"] * I2[i]
            + P["beta"]
            * X
            * S2[i]
            * (
                ((P["k"] * (1 + P["e"])) / 2) * del_2
                + ((P["k"] * (1 - P["e"])) / 2) * del_1
            )
        )
        R2[i + 1] = R2[i] + dt * (P["gamma"] * I2[i])

        S1_sum[i + 1] = np.sum(S1[i + 1] * P["g1"])
        I1_sum[i + 1] = np.sum(I1[i + 1] * P["g1"])
        R1_sum[i + 1] = np.sum(R1[i + 1] * P["g1"])
        S2_sum[i + 1] = np.sum(S2[i + 1] * P["g2"])
        I2_sum[i + 1] = np.sum(I2[i + 1] * P["g2"])
        R2_sum[i + 1] = np.sum(R2[i + 1] * P["g2"])

        incident_inf[i, 0] = np.sum(P["g1"] * (S1[i + 1] - S1[i]) / dt)
        incident_inf[i, 1] = np.sum(P["g2"] * (S2[i + 1] - S2[i]) / dt)

    return [S1_sum, I1_sum, R1_sum, S2_sum, I2_sum, R2_sum, del_ls, incident_inf]


def basic_SIR(P, T, dt, R0):
    t = np.arange(0.0, T + dt, dt, dtype=float)
    num_steps = len(t)

    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)

    S[0] = 0.99
    I[0] = 0.01
    R[0] = 0.00

    beta = P["beta"]
    gamma = beta / R0
    k = P["k"]

    incident_inf = np.empty(num_steps - 1)

    for i in range(num_steps - 1):
        S[i + 1] = S[i] + dt * (-beta * S[i] * k * I[i])
        I[i + 1] = I[i] + dt * (beta * S[i] * k * I[i] - gamma * I[i])
        R[i + 1] = R[i] + dt * (gamma * I[i])

        incident_inf[i] = (S[i + 1] - S[i]) / dt

    return [S, I, R, incident_inf]


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
