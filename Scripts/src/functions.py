import matplotlib as mplb
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.stats import gamma, multivariate_normal
import networkx as nx
import xgi

def create_comm(rho,me=1.5,md=1.5,vare=1,vard=1,samplesize = None):
    """
    Docstring for create_comm

    :param rho: Description
    """
    x = np.linspace(0.1, 3, 100)
    y = np.linspace(0.1, 3, 100)

    X, Y = np.meshgrid(x, y)

    mean_x = me
    mean_y = md

    std_x = vare**0.5
    std_y = vard**0.5

    cov_xy = rho * std_x * std_y

    # 3D matrix with each slice being X and Y, so all the combos
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Defining the mean and cov matrix for our multivariate
    mean = [mean_x, mean_y]
    covariance_matrix = [[std_x**2, cov_xy], [cov_xy, std_y**2]]

    rv = multivariate_normal(mean=mean, cov=covariance_matrix, allow_singular=True)

    joint_dist = rv.pdf(pos)
    if samplesize:
        samples = rv.rvs(size=samplesize*5)
        in_bounds = np.all((samples >= x[0]) & (samples <= x[-1]), axis=1)
        final_samples = samples[in_bounds]
        return joint_dist / np.sum(joint_dist), final_samples[:samplesize]
    return joint_dist / np.sum(joint_dist)


def BC_r0(a, b, beta, gamma, k, e):
    err = ((1 + e) * (a - b)) ** 2 + 4 * a * b * (1 - e)**2
    l = (1 + e) * (a + b) / 4 + np.sqrt(max(err, 0)) / 4
    return beta * k * l / gamma

def BC_beta(a, b, R0, gamma, k, e):
    err = ((1 + e) * (a - b)) ** 2 + 4 * a * b * (1 - e)**2
    l = (1 + e) * (a + b) / 4 + np.sqrt(max(err, 0)) / 4
    return (R0 * gamma) / (k * l)

def IBC_r0(a, b, beta, gamma, k, e, r):
    alpha = (1 - (r**2 + (1-r)**2)) / (r**2 + (1-r)**2)
    err = ((1 + alpha * e) * (r * a - (1-r) * b))**2 + (4 * r * (1-r) * a * b * (1-e)**2)
    l = ((1 + alpha * e) * (r * a + (1-r) * b)) / 2 + np.sqrt(max(err, 0)) / 2
    return beta * k * l / gamma

def IBC_beta(a, b, R0, gamma, k, e, r):
    x = k * (1 + ((1 - r**2 - (1-r)**2) / (r**2 + (1-r)**2))*e)
    y = k * (1 - e)
    # err = (c11 * (a + b))**2 - 4*a*b*(c11**2 - c12**2)
    # l = c11 * (a + b) / 2 + np.sqrt(max(err, 0)) / 2
    err = (r*x*a + (1-r)*x*b)**2 - 4*(r*(1-r)*a*b*(x**2-y**2))
    #l = (r*x*a + (1-r)*x*b) / 2 + np.sqrt(max(err, 0)) / 2
    l = (r*x*a + (1-r)*x*b) / 2 + np.sqrt(max(err, 0)) / 2
    return (R0*gamma)/l

def g0(a, b, beta, r0, k, e):
    c11 = (beta * k * (1 + e)) / 2
    c12 = (beta * k * (1 - e)) / 2
    # err = (c11 * (a + b))**2 - 4*a*b*(c11**2 - c12**2)
    # l = c11 * (a + b) / 2 + np.sqrt(max(err, 0)) / 2
    err = (c11 * (a - b)) ** 2 + 4 * a * b * c12**2
    l = c11 * (a + b) / 2 + np.sqrt(max(err, 0)) / 2
    return l / r0

def plot_comm(joint_dist, x, y):
    ax = sns.heatmap(joint_dist, xticklabels=9, yticklabels=9)
    ax.set(xlabel="susceptibility", ylabel="transmisibility")
    ax.set_xticklabels(np.round([x[i] for i in range(0, 100, 9)], 2), rotation=90)
    ax.set_yticklabels(np.round([y[i] for i in range(0, 100, 9)], 2), rotation=0)
    ax.xaxis.tick_top()
    plt.show()

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

        # del_ls.append([del_1, del_2])

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

def sbm(n, k, epsilon, seed=None):
    """
    Generates a Stochastic Block Model (SBM) graph.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    k : int
        The average degree of each node.
    epsilon : float
        The parameter controlling the ratio of inter- to intra-community edges.
    seed : int, optional
        The seed for the random number generator. Defaults to None.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the generated SBM graph.

    Raises
    ------
    None

    """
    p = k / (n - 1)
    # ratio of inter- to intra-community edges
    p_in = (1 + epsilon) * p
    p_out = (1 - epsilon) * p
    G = nx.planted_partition_graph(2, int(n / 2), p_in, p_out, seed=seed)
    G.add_nodes_from(range(n))
    return nx.adjacency_matrix(G).todense()


    
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
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
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def uniform_HPPM(n, m, k, epsilon, rho=0.5, seed=None):
    r"""Construct the m-uniform hypergraph planted partition model (m-HPPM)

    This uses a fast method for generating hyperedges
    so that instead of the algorithm being of complexity
    :math:`\mathcal{O}(N^m)`, it can be as fast as
    :math:`\mathcal{O}(m(N + |E|))`. See the references
    for more details.

    Parameters
    ----------
    n : int > 0
        Number of nodes
    m : int > 0
        Hyperedge size
    k : float > 0
        Mean degree
    epsilon : float > 0
        Imbalance parameter
    rho : float between 0 and 1, optional
        The fraction of nodes in community 1, default 0.5
    seed : integer or None (default)
        The seed for the random number generator

    Returns
    -------
    Hypergraph
        The constructed m-HPPM hypergraph.

    Raises
    ------
    XGIError
        - If rho is not between 0 and 1
        - If the mean degree is negative.
        - If epsilon is not between 0 and 1

    See Also
    --------
    uniform_HSBM

    Notes
    -----
    Because XGI only stores edges as sets, when self-loops occur,
    they become smaller edges (for example, the edge (0, 0, 0)
    will be mapped to {0}). However, because this is explicitly
    a *uniform* method, we discard these edges so that this is the case.
    For sparse networks, this is a rare occurrence and this method offers
    an order of magnitude speedup.

    References
    ----------
    Nicholas W. Landry and Juan G. Restrepo,
    "Opinion disparity in hypergraphs with community structure",
    Phys. Rev. E **108**, 034311 (2024).
    https://doi.org/10.1103/PhysRevE.108.034311
    """

    if rho < 0 or rho > 1:
        raise xgi.XGIError("The value of rho must be between 0 and 1")
    if k < 0:
        raise xgi.XGIError("The mean degree must be non-negative")

    sizes = [int(rho * n), n - int(rho * n)]

    p = k / (m * n ** (m - 1))
    # ratio of inter- to intra-community edges
    q = rho**m + (1 - rho) ** m
    r = 1 / q - 1
    p_in = (1 + r * epsilon) * p
    p_out = (1 - epsilon) * p

    p = p_out * np.ones([2] * m)
    p[tuple([0] * m)] = p_in
    p[tuple([1] * m)] = p_in

    return xgi.uniform_HSBM(n, m, p, sizes, seed=seed)