# This module contains all of the generative network models
import networkx as nx
import xgi


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
