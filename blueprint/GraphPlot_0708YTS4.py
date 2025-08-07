import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def read_PipeFile():
    """Read pipeline data from the local file"""
    with open('0708YTS4.txt', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Read nodes and edges
    nodelist = data['nodelist']
    edgelist = data['linklist']
    
    # Convert to DataFrames
    nodeDF = pd.DataFrame(nodelist)
    edgeDF = pd.DataFrame(edgelist)
    
    # Use original IDs
    nodeID = nodeDF['id']
    edgeID = edgeDF['id']
    
    # Store original edge endpoints
    edgeDF['sourceid_original'] = edgeDF['sourceid']
    edgeDF['targetid_original'] = edgeDF['targetid']

    # Extract node positions and types
    nodeParas = []
    nodePSTs = []
    nodeTypes = []
    
    for i in range(len(nodeDF)):
        try:
            if isinstance(nodeDF['parameter'].iloc[i], str):
                para = json.loads(nodeDF['parameter'].iloc[i])
            else:
                para = nodeDF['parameter'].iloc[i]
            nodeParas.append(para)
            
            # Extract position information
            if 'styles' in para and 'position' in para['styles']:
                position = para['styles']['position']
                nodePSTs.append([position['x'], position['y']])
            else:
                nodePSTs.append([0, 0])
                print(f"Warning: Node {nodeDF['id'].iloc[i]} has no position info, using (0, 0)")
            
            # Extract node type
            node_type = para.get('type', 'Unknown')
            nodeTypes.append(node_type)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing node {nodeDF['id'].iloc[i]} parameters: {e}")
            nodeParas.append({})
            nodePSTs.append([0, 0])
            nodeTypes.append('Unknown')
    
    nodePSTs = pd.Series(nodePSTs)
    nodeTypes = pd.Series(nodeTypes)

    return nodeDF, edgeDF, nodeID, edgeID, nodePSTs, nodeTypes

def make_edge_attr(edgeDF, nodeID_to_index):
    """Create edge attributes with length information"""
    edge_data = []
    
    for i in range(len(edgeDF)):
        source_id = edgeDF['sourceid_original'].iloc[i]
        target_id = edgeDF['targetid_original'].iloc[i]
        
        # Check if nodes exist
        if source_id in nodeID_to_index and target_id in nodeID_to_index:
            from_idx = nodeID_to_index[source_id]
            to_idx = nodeID_to_index[target_id]
            
            # Parse edge parameters to get length
            try:
                if isinstance(edgeDF['parameter'].iloc[i], str):
                    edge_para = json.loads(edgeDF['parameter'].iloc[i])
                else:
                    edge_para = edgeDF['parameter'].iloc[i]
                
                # Get length information
                length = edge_para.get('parameter', {}).get('Length', 1.0)
                if length is None or length <= 0:
                    length = 1.0
                    
                edge_data.append({
                    'from': from_idx,
                    'to': to_idx,
                    'length': float(length),
                    'edge_id': edgeDF['id'].iloc[i]
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing edge {edgeDF['id'].iloc[i]} parameters: {e}, using default length 1.0")
                edge_data.append({
                    'from': from_idx,
                    'to': to_idx,
                    'length': 1.0,
                    'edge_id': edgeDF['id'].iloc[i]
                })
        else:
            print(f"Warning: Edge {edgeDF['id'].iloc[i]} endpoints not found, skipping")
    
    return pd.DataFrame(edge_data)

def get_node_visual_properties(node_type):
    """Get enhanced visual properties for different node types"""
    type_properties = {
        'VavlePro': {'color': '#E74C3C', 'marker': 's', 'size': 120, 'edge_color': '#C0392B'},
        'Stream': {'color': '#3498DB', 'marker': 'o', 'size': 100, 'edge_color': '#2980B9'},
        'Tee': {'color': '#2ECC71', 'marker': '^', 'size': 110, 'edge_color': '#27AE60'},
        'Mixer': {'color': '#F39C12', 'marker': 'D', 'size': 105, 'edge_color': '#E67E22'},
        'Pipe': {'color': '#9B59B6', 'marker': 'h', 'size': 80, 'edge_color': '#8E44AD'},
        'Unknown': {'color': '#95A5A6', 'marker': 'o', 'size': 70, 'edge_color': '#7F8C8D'}
    }
    return type_properties.get(node_type, type_properties['Unknown'])

def create_real_coordinate_visualization(show_edge_labels=False, label_threshold=500, 
                                       figure_size=(24, 18), edge_alpha=0.6,
                                       improve_spacing=True):
    """Create visualization using REAL coordinates from the blueprint
    
    Args:
        show_edge_labels (bool): Whether to show edge length labels
        label_threshold (float): Only show labels for edges shorter than this length
        figure_size (tuple): Figure size in inches
        edge_alpha (float): Edge transparency
        improve_spacing (bool): Whether to apply minimal spacing improvements
    """
    
    # Read data
    print("Loading pipeline data...")
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs, nodeTypes = read_PipeFile()
    
    print(f"Loaded {len(nodeDF)} nodes and {len(edgeDF)} edges")
    
    # Create node ID mapping
    nodeID_to_index = {val: idx for idx, val in enumerate(nodeID)}
    
    # Create edge attributes
    edge_attr = make_edge_attr(edgeDF, nodeID_to_index)
    
    if len(edge_attr) == 0:
        print("Error: No valid edge data")
        return
    
    print(f"Processed {len(edge_attr)} valid edges")
    
    # Create DIRECTED graph
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodeID)))
    
    # Add edges with length attributes
    for _, edge in edge_attr.iterrows():
        G.add_edge(edge['from'], edge['to'], length=edge['length'])
    
    print(f"Directed graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Use REAL coordinates directly from the blueprint
    pos = {i: nodePSTs.iloc[i] for i in range(len(nodePSTs))}
    
    # Display coordinate range information
    all_coords = np.array(list(pos.values()))
    x_coords = all_coords[:, 0]
    y_coords = all_coords[:, 1]
    
    print(f"\n📍 REAL COORDINATE RANGES:")
    print(f"X: {x_coords.min():.0f} to {x_coords.max():.0f} (range: {x_coords.max()-x_coords.min():.0f})")
    print(f"Y: {y_coords.min():.0f} to {y_coords.max():.0f} (range: {y_coords.max()-y_coords.min():.0f})")
    
    # Optional: Apply minimal spacing improvement while preserving real coordinates
    if improve_spacing:
        print("Applying minimal spacing improvements...")
        min_distance = 200  # Minimum distance in real coordinate units
        pos_array = np.array(list(pos.values()))
        
        # Simple repulsion to avoid very close nodes
        for iteration in range(10):  # Limited iterations to preserve real positions
            for i in range(len(pos_array)):
                for j in range(i + 1, len(pos_array)):
                    distance = np.linalg.norm(pos_array[i] - pos_array[j])
                    if 0 < distance < min_distance:
                        # Apply small repulsion
                        direction = (pos_array[i] - pos_array[j]) / distance
                        adjustment = (min_distance - distance) * 0.05  # Small adjustment
                        pos_array[i] += direction * adjustment
                        pos_array[j] -= direction * adjustment
        
        pos = {i: pos_array[i] for i in range(len(pos_array))}
    
    # Create enhanced visualization
    fig, ax = plt.subplots(figsize=figure_size, facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Get unique node types
    unique_types = nodeTypes.unique()
    print(f"Node types found: {list(unique_types)}")
    
    # Draw edges first
    print("Drawing edges with real coordinate relationships...")
    edge_count = 0
    for _, edge in edge_attr.iterrows():
        from_pos = pos[edge['from']]
        to_pos = pos[edge['to']]
        
        # Calculate real distance
        real_distance = np.linalg.norm(np.array(to_pos) - np.array(from_pos))
        
        # Color based on real geometric distance
        color_intensity = min(1.0, max(0.3, 1.0 - (real_distance / 10000.0)))
        edge_color = plt.cm.Blues(color_intensity)
        
        # Draw edge with arrow
        ax.annotate('', xy=to_pos, xytext=from_pos,
                   arrowprops=dict(arrowstyle='->', 
                                 color=edge_color, 
                                 alpha=edge_alpha, 
                                 lw=1.0,
                                 shrinkA=6, shrinkB=6))
        edge_count += 1
    
    # Draw nodes by type
    print("Drawing nodes at real coordinates...")
    for node_type in unique_types:
        nodes_of_type = [i for i, t in enumerate(nodeTypes) if t == node_type]
        if not nodes_of_type:
            continue
            
        props = get_node_visual_properties(node_type)
        
        node_positions = [pos[node] for node in nodes_of_type]
        x_coords = [p[0] for p in node_positions]
        y_coords = [p[1] for p in node_positions]
        
        ax.scatter(x_coords, y_coords, 
                  c=props['color'], 
                  marker=props['marker'], 
                  s=props['size'],
                  label=f'{node_type} ({len(nodes_of_type)})',
                  alpha=0.8, 
                  edgecolors=props['edge_color'], 
                  linewidth=1.5,
                  zorder=5)
    
    # Enhanced plot customization
    ax.set_title('Graph Structure of YanTai_S4 Steam Pipeline Network', 
                fontsize=18, fontweight='bold', pad=20)
    #ax.set_xlabel('X', fontsize=14)
    #ax.set_ylabel('Y', fontsize=14)
    
    # Legend
    legend = ax.legend(title='Node Types', 
                      bbox_to_anchor=(1.02, 1), 
                      loc='upper left',
                      frameon=True,
                      fancybox=True,
                      shadow=True)
    
    # Grid and formatting
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'YTS4_graph_structure.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    
    # Display statistics
    print(f"\n{'='*50}")
    print("    REAL COORDINATE VISUALIZATION STATS")
    print("="*50)
    print(f"📊 Total nodes: {len(G.nodes())}")
    print(f"🔗 Total edges: {len(G.edges())}")
    print(f"📐 Coordinate system: REAL blueprint coordinates")
    print(f"🎯 Spacing improvements: {'Applied' if improve_spacing else 'None'}")
    
    print(f"\n✅ Real coordinate visualization saved as '{output_filename}'")
    
    return G, pos, nodeTypes, edge_attr

if __name__ == '__main__':
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    print("🎯 Creating REAL COORDINATE Visualization...")
    print("="*50)
    
    # Create visualization with real coordinates
    G, pos, nodeTypes, edge_attr = create_real_coordinate_visualization(
        show_edge_labels=False,
        improve_spacing=True,  # Set to False for absolutely no coordinate changes
        figure_size=(20, 16),
        edge_alpha=0.7
    )
    
    print("\n🎉 Real coordinate visualization complete!")
