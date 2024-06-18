import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def create_map_and_nodes(grid_size, node_list_size):
    """Create a 'grid_size' map (grid) of random signal qualities between 0 and 1."""
    signal_quality = np.random.rand(grid_size, grid_size)
    coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    np.random.shuffle(coords)
    selected_coords = coords[:node_list_size]
        
    node_list = []
    for index, coord in enumerate(selected_coords):
        x, y = coord
        node_list.append(
            (x, y, signal_quality[x, y])
        )
    return signal_quality, node_list


def plot(signal_quality, route, filename=None):
    """Make transformations on route to print results onto signal_quality grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(signal_quality, cmap="viridis", origin='upper', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Signal Quality')

    # add text values to each grid tile
    grid_size = signal_quality.shape[0]
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{signal_quality[i, j]:.2f}", ha='center', va='center', color='black')

    positions = {}
    G = nx.DiGraph()
    for edge in route.edges:
        G.add_edge(edge.origin.id_, edge.end.id_, weight=edge.cost)
        positions[edge.origin.id_] = (edge.origin.y, edge.origin.x)
        positions[edge.end.id_] = (edge.end.y, edge.end.x)

    # now we can paint the position of each node
    node_colors = ['lightgreen' if node == route.edges[0].origin.id_ else 'skyblue' for node in G.nodes()]
    nx.draw(G, positions, node_size=500, node_color=node_colors, arrows=True, edge_color='white', ax=ax)

    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    plt.title('Signal Quality Grid')
    if filename is not None:
        plt.savefig(filename)
    else: 
        plt.show()
