import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys
from pyvis.network import Network

def plot_narrative_graph(nodes, edges, figsize=(12, 8)):
    """
    Plot a narrative graph with edges colored according to timesteps.
    Only plots nodes that have connections.
    
    Args:
        nodes: List of node identifiers
        edges: List of tuples (source, target, (relation_type, timestep))
        figsize: Tuple of (width, height) for the plot
    """
    # Create graph
    G = nx.DiGraph()
    
    # Get all connected nodes from edges
    connected_nodes = set()
    for s, t, _ in edges:
        connected_nodes.add(s)
        connected_nodes.add(t)
    
    # Only add nodes that are connected
    G.add_nodes_from(connected_nodes)
    
    # Separate edge information
    edge_list = [(s, t) for s, t, _ in edges]
    edge_types = [e[2][0] for e in edges]
    timesteps = [e[2][1] for e in edges]
    
    # Add edges to graph
    G.add_edges_from(edge_list)
    
    # Create color map based on timesteps
    norm = plt.Normalize(min(timesteps), max(timesteps))
    cmap = plt.cm.winter
    edge_colors = [cmap(norm(t)) for t in timesteps]
    
    # Create the plot with proper axes setup
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the layout with more space between nodes
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    # Draw edges with colors
    for (s, t), color in zip(edge_list, edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[(s, t)],
                             edge_color=[color],
                             arrows=True,
                             arrowsize=20,
                             node_size=2000,
                             ax=ax,
                             connectionstyle="arc3,rad=0.2")  # Curved edges
    
    # Add edge labels
    edge_labels = {(s, t): f"{rel_type}" 
                  for (s, t, (rel_type, time)) in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                               ax=ax, font_size=7)
    
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


def plot_draggable_graph(nodes, edges, filename='draggable_graph.html', height="750px", width="100%"):
    """
    Create an interactive graph visualization with draggable nodes using pyvis.
    
    Args:
        nodes: List of node identifiers
        edges: List of tuples (source, target, (relation_type, timestep))
        filename: Path to save the HTML file
        height: Height of the graph (default: "750px")
        width: Width of the graph (default: "100%")
    """
    # Create network
    net = Network(height=height, width=width, bgcolor="#ffffff", 
                 font_color="black", directed=True)
    
    # Get all connected nodes from edges
    connected_nodes = set()
    for s, t, _ in edges:
        connected_nodes.add(s)
        connected_nodes.add(t)
    
    # Add nodes
    for node in connected_nodes:
        net.add_node(node, label=node, color='lightblue', 
                    title=node)
    
    # Get timesteps for color scaling
    timesteps = [e[2][1] for e in edges]
    min_time, max_time = min(timesteps), max(timesteps)
    
    # Create color map based on timesteps (using winter colormap)
    norm = plt.Normalize(min_time, max_time)
    cmap = plt.cm.winter
    
    # Add edges with colors based on timesteps
    for source, target, (relation, time) in edges:
        # Get color from winter colormap
        rgba_color = cmap(norm(time))
        # Convert RGBA to hex
        hex_color = f'#{int(rgba_color[0]*255):02x}{int(rgba_color[1]*255):02x}{int(rgba_color[2]*255):02x}'
        
        net.add_edge(source, target, 
                    title=f"{relation} (Time: {time})",  # hover text
                    label=relation,
                    color=hex_color,
                    arrows='to')
    
    # Add physics layout options for better visualization
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "edges": {
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        }
    }
    """)
    
    # Save the graph
    net.save_graph(filename)

if __name__ == "__main__":
    # Example data
    nodes = [
    'Indiana Jones', 'Belloq', 'Marion', 'Toht', 
    'Sallah', 'Hitler', 'US Intelligence Agents',
    'Golden Idol', 'Medallion', 'Ark of the Covenant', 
    'Staff of Ra', 
    'Peruvian Temple', 'Nepal Bar', 'Cairo', 'Well of Souls', 
    'Aegean Island', 'Washington DC'
    ]

    edges = [
        ('Indiana Jones', 'Golden Idol', ('discovers', 1)),
        ('Belloq', 'Golden Idol', ('steals', 2)),
        ('US Intelligence Agents', 'Indiana Jones', ('recruits', 3)),
        ('Indiana Jones', 'Marion', ('reunites', 4)),
        ('Toht', 'Medallion', ('attempts_to_steal', 5)),
        ('Indiana Jones', 'Medallion', ('obtains', 6)),
        ('Sallah', 'Indiana Jones', ('helps', 7)),
        ('Indiana Jones', 'Ark of the Covenant', ('seeks', 8)),
        ('Belloq', 'Nazi Forces', ('assists', 9)),
        ('Indiana Jones', 'Well of Souls', ('discovers', 10)),
        ('Belloq', 'Ark of the Covenant', ('seizes', 11)),
        ('Marion', 'Ark of the Covenant', ('captive_with', 12)),
        ('Belloq', 'Ark of the Covenant', ('opens', 13)),
        ('Ark of the Covenant', 'Nazi Forces', ('destroys', 14)),
        ('US Intelligence Agents', 'Ark of the Covenant', ('stores', 15))
    ]

    #Save plots
    save_narrative_graph(nodes, edges, 'narrative_graph.png')
    plot_draggable_graph(nodes, edges, 'draggable_narrative_graph.html') 