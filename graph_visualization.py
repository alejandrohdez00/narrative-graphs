import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys

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
    
    # Only add nodes that appear in edges
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
    
    # Add edge labels without timesteps
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

def plot_interactive_graph(nodes, edges, figsize=(1200, 800)):
    """
    Create an interactive plot of the narrative graph using Plotly.
    
    Args:
        nodes: List of node identifiers
        edges: List of tuples (source, target, (relation_type, timestep))
        figsize: Tuple of (width, height) for the plot in pixels
    """
    # Create graph
    G = nx.DiGraph()
    
    # Get all connected nodes from edges
    connected_nodes = set()
    for s, t, _ in edges:
        connected_nodes.add(s)
        connected_nodes.add(t)
    
    # Only add nodes that appear in edges
    G.add_nodes_from(connected_nodes)
    G.add_edges_from([(s, t) for s, t, _ in edges])
    
    # Get timesteps for color scaling
    timesteps = [e[2][1] for e in edges]
    min_time, max_time = min(timesteps), max(timesteps)
    
    # Create layout
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Create edge traces
    edge_traces = []
    edge_labels = []
    
    for edge in edges:
        source, target, (relation, time) = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Calculate the control point for the curved edge
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # Add some curvature
        curve_x = mid_x + (y1 - y0) * 0.2
        curve_y = mid_y - (x1 - x0) * 0.2
        
        # Calculate color based on timestep
        color_val = (time - min_time) / (max_time - min_time)
        rgb = colorsys.hsv_to_rgb(0.6 * (1 - color_val), 0.8, 0.8)
        color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
        
        # Create edge trace with curved path
        edge_trace = go.Scatter(
            x=[x0, curve_x, x1],
            y=[y0, curve_y, y1],
            line=dict(width=2, color=color),
            hoverinfo='text',
            text=f"{source} {relation} {target}<br>Time: {time}",
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Add arrow at the end of the edge
        arrow_trace = go.Scatter(
            x=[curve_x, x1, curve_x],
            y=[curve_y, y1, curve_y],
            line=dict(width=0.5, color=color),
            fill='toself',
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(arrow_trace)
        
        # Add edge label at control point
        edge_labels.append(
            go.Scatter(
                x=[curve_x],
                y=[curve_y],
                text=[relation],
                mode='text',
                textposition='middle center',
                hoverinfo='none',
                showlegend=False,
                textfont=dict(size=10)
            )
        )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            size=40,
            color='lightblue',
            line=dict(width=2, color='black')
        )
    )

    # Create figure
    fig = go.Figure(
        data=[*edge_traces, *edge_labels, node_trace],
        layout=go.Layout(
            title='Interactive Narrative Graph',
            titlefont_size=16,
            showlegend=False,
            width=figsize[0],
            height=figsize[1],
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    return fig

def save_interactive_graph(nodes, edges, filename, figsize=(1200, 800)):
    """
    Save the interactive graph visualization as HTML file.
    
    Args:
        nodes: List of node identifiers
        edges: List of tuples (source, target, (relation_type, timestep))
        filename: Path to save the HTML file
        figsize: Tuple of (width, height) for the plot in pixels
    """
    fig = plot_interactive_graph(nodes, edges, figsize)
    fig.write_html(filename)

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

    # Create and save both static and interactive visualizations
    plot_narrative_graph(nodes, edges)
    plt.show()
    save_narrative_graph(nodes, edges, 'narrative_graph.png')
    
    # Create and save interactive visualization
    save_interactive_graph(nodes, edges, 'interactive_narrative_graph.html') 