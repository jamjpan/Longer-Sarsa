'''
Use SARSA to discover a longest path in given graph.
'''
import networkx as nx
import matplotlib.pyplot as plt
import random

# Set the graph side-length.
N = 4

# Set how many episodes you want?
episodes = 10000

# Set the epsilon for e-greedy action selection.
epsilon = 0.1

# Set the step size for updating action-value function.
alpha = 0.1

# Select which graph you want? Also set the source U and target V
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(N, N))
# U = 0
# V = (N*N - 1)

G = nx.convert_node_labels_to_integers(nx.navigable_small_world_graph(N))
U = 0
V = (N*N - 1)

################################################################################
def clear_render():
    plt.clf()
    nx.draw_networkx(G, pos=pos)
    plt.draw()


def is_valid(g):  # returns a boolean. g is any subgraph of G.
    # If not a simple path, then g is "not valid".
    if not nx.is_simple_path(G, list(g.nodes)):
        return False

    # If source vertex U not in g, then g is not valid.
    if not (U in g.nodes):
        return False

    # If target vertex V not in g, then g is not valid.
    if not (V in g.nodes):
        return False

    # Case A: G is undirected (hence g is undirected)
    if type(g) == nx.Graph:

        # If source vertex U has more than 1 neighbor, then g is not valid.
        if not (len(list(g.neighbors(U))) == 1):
            return False

        # If target vertex V has more than 1 neighbor, then g is not valid.
        if not (len(list(g.neighbors(V))) == 1):
            return False

    # Case B: G is directed (hence g is directed)
    if type(g) == nx.DiGraph:

        # If source vertex U has predecessor (incoming edge), then g not valid.
        if not (len(list(g.predecessors(U))) == 0):
            return False

        # If target vertex V has successor (outgoing edge), then g not valid.
        if not (len(list(g.successors(V))) == 0):
            return False

    return True


def find_id(g):
    global sid
    for key in lu_sid:
        if lu_sid[key].edges == g.edges:
            return key
    sid += 1
    return sid


def find_tip(g):
    # If g has only one vertex, the "tip" is this vertex
    if len(g) == 1:
        return list(g.nodes)[0]

    # If g is undirected, the tip is the non-source vertex with one neighbor
    if type(g) == nx.Graph:
        for n in g.nodes:
            if n != U and len(list(g.neighbors(n))) == 1:
                return n

    # If g is directed, the tip is the vertex with zero successors
    if type(g) == nx.DiGraph:
        for n in g.nodes:
            if n != U and len(list(g.successors(n))) == 0:
                return n


def fill_actions(s):  # s is the id of one specific state (subgraph of G)
    global A_s
    if s not in A_s:
        A_s[s] = []
        g = lu_sid[s]
        if not is_valid(g):
            i = find_tip(g)
            # for DiGraph, it can happen that we grew the path "backwards" and
            # arrived at the source. In this case there are no actions..
            if i == None:
                return
            for k in G.neighbors(i):
                if not g.has_node(k):
                    A_s[s].append((i, k))  # means: draw edge from i ~> k
                    #print("discovered add({}, {})".format(i, k))


def select_action(s):  # s is the id of one specific state (subgraph of G)
    if s in Q_s_a:
        # Greedily select next action with probability 1-epsilon
        return (max(Q_s_a[s], key=Q_s_a[s].get)
            if random.random() > epsilon else random.choice(A_s[s]))
    else:
        fill_actions(s)
        return random.choice(A_s[s]) if len(A_s[s]) > 0 else None


def play():
    global lu_sid
    global Q_s_a

    # Initialize the first state and get action according to policy
    ep_G0 = G0.copy()
    ep_S0 = find_id(ep_G0)
    ep_A0 = select_action(ep_S0)
    reward = 0

    while (not is_valid(ep_G0)) and (ep_A0 is not None):
        # Perform action and collect reward
        ep_G1 = ep_G0.copy()
        ep_G1.add_edge(ep_A0[0], ep_A0[1])
        ep_S1 = find_id(ep_G1)
        reward = ep_G1.size() if is_valid(ep_G1) else 0

        # (Maintain our lookup table)
        if ep_S1 not in lu_sid:
            lu_sid[ep_S1] = ep_G1.copy()

        # Select the next action on ep_S1
        ep_A1 = select_action(ep_S1)

        # Update Q_s_a
        if ep_S0 not in Q_s_a:
            Q_s_a[ep_S0] = {}
        q0 = Q_s_a[ep_S0][ep_A0] if ep_S0 in Q_s_a and ep_A0 in Q_s_a[ep_S0] else 0
        q1 = Q_s_a[ep_S1][ep_A1] if ep_S1 in Q_s_a and ep_A1 in Q_s_a[ep_S1] else 0
        qn = q0 + alpha*(reward + q1 - q0)
        Q_s_a[ep_S0][ep_A0] = qn

        # Prepare for next state step
        ep_G0 = ep_G1.copy()
        ep_S0 = find_id(ep_G0)
        ep_A0 = ep_A1;

    return reward


def train():
    for i in range(0, episodes):
        reward = play()
        print("ep={:04d}, return={}".format(i, reward))


def test():
    my_g = G0.copy()

    while not is_valid(my_g):
        print(list(my_g.edges))

        my_s = find_id(my_g)
        if my_s not in Q_s_a:
            print("Never learned this state!")
            return

        my_a = max(Q_s_a[my_s], key=Q_s_a[my_s].get)
        my_g.add_edge(my_a[0], my_a[1])

        clear_render()
        nx.draw_networkx_edges(G, pos=pos, edgelist=my_g.edges,
            edge_color='r', width=2)
        plt.pause(0.5)

    print("Target reached. Score={}".format(len(list(my_g.edges))))
    return


################################################################################
sid = 0
pos = nx.spectral_layout(G)

# Uncomment if want to preview the graph
# nx.draw_networkx(G, pos=pos)
# plt.draw()
# plt.show()

G0 = G.subgraph([U])

# Each state (a nx.Graph or nx.DiGraph representing a path through G) is hashed
# to a unique ID, and we use lu_sid to remember these IDs.
lu_sid = { sid: G0.copy() }

# Initialize the action-value function results.
Q_s_a = {}  # { sid -> { action -> return } }

# Remember which actions are available for which state.
A_s = {}  # { sid -> [ (i, k)... ] }

print("Start")
print("Source={}, Target={}".format(U, V))

# Estimate the action-value function
train()

# Follow the greedy learned action-values to get the longest path
test()

print("Finish")
# for s in Q_s_a:
#     print("State {} ({})".format(s, lu_sid[s].edges()))
#     for a in Q_s_a[s]:
#         print("  {}: {}".format(a, Q_s_a[s][a]))
plt.show()

