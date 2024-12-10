import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_narrative_graph(nodes, edges, figsize=(12, 8)):
    """
    Plot a narrative graph with edges colored according to timesteps.
    
    Args:
        nodes: List of node identifiers
        edges: List of tuples (source, target, (relation_type, timestep))
        figsize: Tuple of (width, height) for the plot
    """
    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    # Separate edge information first
    edge_list = [(s, t) for s, t, _ in edges]
    edge_types = [e[2][0] for e in edges]
    timesteps = [e[2][1] for e in edges]
    
    # Now add edges to graph
    G.add_edges_from(edge_list)
    
    # Create color map based on timesteps
    norm = plt.Normalize(min(timesteps), max(timesteps))
    cmap = plt.cm.winter
    edge_colors = [cmap(norm(t)) for t in timesteps]
    
    # Create the plot with proper axes setup
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the layout
    pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Draw edges with colors
    for (s, t), color in zip(edge_list, edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[(s, t)],
                             edge_color=[color],
                             arrows=True,
                             arrowsize=20,
                             ax=ax)
    
    # Add edge labels
    edge_labels = {(s, t): f"{rel_type}\n(t={time})" 
                  for (s, t, (rel_type, time)) in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Timestep')
    
    plt.title("Narrative Graph")
    ax.set_axis_off()
    
    return plt

def save_narrative_graph(nodes, edges, filename, figsize=(12, 8)):
    """
    Save the narrative graph visualization to a file.
    
    Args:
        nodes: List of node identifiers
        edges: List of tuples (source, target, (relation_type, timestep))
        filename: Path to save the image
        figsize: Tuple of (width, height) for the plot
    """
    plt = plot_narrative_graph(nodes, edges, figsize)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example data
    nodes = ['Event A', 'Event B', 'Event C', 'Event D']
    edges = [
        ('Event A', 'Event B', ('causes', 1)),
        ('Event B', 'Event C', ('follows', 2)),
        ('Event A', 'Event C', ('influences', 3)),
        ('Event C', 'Event D', ('leads_to', 4))
    ]

    # Display the graph
    plot_narrative_graph(nodes, edges)
    plt.show()

    # Save the graph
    save_narrative_graph(nodes, edges, 'narrative_graph.png') 