# class to visualize flow network
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd
import numpy as np


class FlowViz():

    def __init__(self, graph):
        """
        initializes the FlowViz object with a graph
        :param graph: FlowGraph object to be visualized
        """
        self.graph = graph            


    def plot_flowgraph(self, edge_width_field='flow', edge_color_field='utilisation', colormap_name='RdYlGn_r', 
                   node_size_type='max_flow', min_width=0.25, max_width=5):
        """
        plot nodes and edges of graph with flow and utilisation information
        :param edge_width_field: field of edge object to be used as width of edge
        :param edge_color_field: field of edge object to be used as color of edge
        :param colormap_name: name of colormap to be used for edge colors
        :param node_size_type: field of node object to be used as size of node (if "max_flow" the maximum of inflow and 
                               outflow is used)
        :param min_width: minimum width of edge
        :param max_width: maximum width of edge
        :return: plotly figure object
        """
        # set plot properties
        self.edge_width_field = edge_width_field
        self.edge_color_field = edge_color_field
        self.colormap_name = colormap_name
        self.node_size_type = node_size_type
        self.min_width = min_width
        self.max_width = max_width
        
        # get nodes and edges
        self.node_df = self.graph.nodes_to_df().set_index('name')
        self.edge_df = self.graph.edges_to_df().set_index(['origin', 'destination'])

        # set edge and node colores
        self.edge_colormap = plt.get_cmap(self.colormap_name)
        self.node_colors = self._get_color_dict(self.node_df['type'])

        # set sizes
        self.xy_range = self.node_df[['x', 'y']].apply([min, max]).diff().loc['max'].min()       
        if node_size_type == 'max_flow': 
            node_size = self.node_df[['inflow', 'outflow']].max(axis=1)
            self.node_size_range = node_size.apply([min, max])
        elif node_size_type in self.node_df.columns:
            self.node_size_range = self.node_df[node_size_type].apply([min, max])
        else: 
            self.node_size_range = pd.Series({'min': 0, 'max': 1})
        self.width_range = self.edge_df[self.edge_width_field].apply([min, max])

        # create nodes
        node_traces = []
        node_shapes = []
        annotations = []
        for node_name, node in self.node_df.iterrows():
            shape, trace, annotation = self._create_node_object(node_name, node)
            node_traces.append(trace)
            node_shapes.append(shape)
            annotations.append(annotation)        
        
        # create edges
        edge_traces = []
        for (origin_name, destination_name), edge in self.edge_df.iterrows():
            origin = self.node_df.loc[origin_name]
            destination = self.node_df.loc[destination_name]
            edge_traces.extend(self._create_edge_object(edge, origin, destination))

        # create colorbar
        colorbar_trace = self._create_colorbar_object()

        # create legend
        legend_traces = self._create_legend_object()

        # create figure
        fig = go.Figure()
        fig.update_layout(hovermode='closest', 
            xaxis=dict(visible=False),  
            yaxis=dict(visible=False), 
            plot_bgcolor='rgba(0,0,0,0)',  
            paper_bgcolor='rgba(0,0,0,0)', 
            legend=dict(
                x=self.node_df['x'].min() - self.xy_range * 0.15,  
                y=self.node_df['y'].max() + self.xy_range * 0.15,
                traceorder='normal',
                font=dict(size=12, color='white'),
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=1), 
            shapes=node_shapes, 
            annotations=annotations)   
        fig.add_traces(node_traces + edge_traces + legend_traces)
        # check if colorbar shall be added
        if colorbar_trace: fig.add_traces(colorbar_trace)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig
    
    def plot_graph(self):
        """
        plot nodes and edges of graph only
        :return: plotly figure object
        """
        return self.plot_flowgraph(edge_width_field='capacity', node_size_type=None, edge_color_field=None)
    
    def plot_sankey(self, link_width_field='flow', link_color_field='utilisation', colormap_name='RdYlGn_r'):
        """
        plot sankey diagram of graph of optimised flow
        :param link_width_field: field of edge object to be used as width of link
        :param link_color_field: field of edge object to be used as color of link
        :param colormap_name: name of colormap to be used for link colors
        :return: plotly figure object
        """        
        # set color map
        self.colormap_name = colormap_name

        # get nodes and edges
        self.node_df = self.graph.nodes_to_df().set_index('name')
        self.edge_df = self.graph.edges_to_df().set_index(['origin', 'destination'])

        node_to_idx = {name: i for i, name in enumerate(self.node_df.index)}
        node_colors = self._get_color_dict(self.node_df['type'])

        fig = go.Figure(go.Sankey(
                node = dict(
                    pad=15,
                    thickness=20,
                    label=self.node_df.index,
                    color=[node_colors[node] for node in self.node_df['type']],
                    customdata=[self._get_node_hover_text(node_name, node) for node_name, node in self.node_df.iterrows()],
                    hovertemplate='%{customdata}<extra></extra>'
                    ),
                link=dict(
                    source=[node_to_idx[origin] for origin, _ in self.edge_df.index],
                    target=[node_to_idx[destination] for _, destination in self.edge_df.index],
                    value=self.edge_df[link_width_field],
                    color=[self._get_edge_color(edge[link_color_field]) for _, edge in self.edge_df.iterrows()],
                    customdata=[self._get_edge_hover_text(edge) for _, edge in self.edge_df.iterrows()],
                    hovertemplate='%{customdata}<extra></extra>'        
                )
            ))

        fig.update_layout(plot_bgcolor='black',  paper_bgcolor='black',  font=dict(color='white'))        

        return fig

    @staticmethod
    def _get_field_hover_text(series):
        """
        generate hover text for series
        :param series: pandas series object
        :return: string with hover text
        """
        text = ''
        for key, value in series.items():
            if value is None or value == float('inf'): continue

            # format value
            if isinstance(value, float):
                if abs(value) >= 100:
                    value_str = f"{value:.0f}"
                elif abs(value) >= 10:
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{value:.2f}"
            elif isinstance(value, int):
                value_str = f"{value}"
            else:
                value_str = value
            
            # format key
            key = key.replace('_', ' ').title()

            # append to text
            text += f"{key}: {value_str}<br>"
        return text        
    
    @staticmethod
    def _get_edge_hover_text(edge):
        """
        generate hover text for edge
        :param edge: edge object
        :return: string with hover text
        """
        title = f"<b>{edge.name[0]} -> {edge.name[1]}</b><br>"
        return title + FlowViz._get_field_hover_text(edge)

    @staticmethod
    def _get_node_hover_text(node_name, node):
        """
        generate hover text for node
        :param node_name: name of node
        :param node: node object
        :return: string with hover text
        """
        title = f"<b>{node_name}</b><br>"
        return title + FlowViz._get_field_hover_text(node)

    @staticmethod
    def _create_vector(x1, y1, x2, y2, l):
        """
        create a vector from x1, y1 in the direction of x2, y2 with length l
        :param x1: x-coordinate of start point
        :param y1: y-coordinate of start point
        :param x2: x-coordinate of end point
        :param y2: y-coordinate of end point
        :param l: length of vector
        :return: tuple with x and y coordinates of vector
        """
        # create vector along (x1, y1) to (x2, y2) with length l
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5

        if length == 0: return 0, 0
        
        return dx / length * l, dy / length * l 

    @staticmethod    
    def _norm_vals(vals, out_min, out_max):
        """
        normalize values to range [out_min, out_max]
        :param vals: numpy array of values
        :param out_min: minimum of output range
        :param out_max: maximum of output range
        :return: numpy array with normalized values
        """
        return out_min + (out_max - out_min) * (vals - vals.min()) / (vals.max() - vals.min())

    def _get_edge_color(self, val):
        """
        generate color for edge based on value
        :param val: value of edge
        :return: rgba color string
        """
        color = self.edge_colormap(val)
        return f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})'

    @staticmethod
    def _get_color_dict(names):
        """
        generate dictionary with colors for unique names
        :param names: list of names
        :return: dictionary with name as key and color as value
        """
        # get unique names and sort
        unique_names = sorted(set(names))
        return {name: DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)] for i, name in enumerate(unique_names)}

    def _get_node_radius(self, node):
        """
        get node radius based on node size
        :param node: node object
        :return: radius of node
        """
        if self.node_size_type == 'max_flow':
            value = max(node['inflow'], node['outflow'])
        elif self.node_size_type in node:
            value = node[self.node_size_type]
        else:
            value = 0.5
        out_min = 0.005 * self.xy_range
        out_max = 0.05 * self.xy_range
        ratio = (value - self.node_size_range['min']) / (self.node_size_range['max'] - self.node_size_range['min'])
        return out_min + (out_max - out_min) * ratio
    
    def _get_edge_width(self, edge):
        """
        get edge width based on edge width field
        :param edge: edge object
        :return: width of edge
        """
        ratio = (edge[self.edge_width_field] - self.width_range['min']) / (self.width_range['max'] - self.width_range['min'])
        return self.min_width + (self.max_width - self.min_width) * ratio

    def _create_node_object(self, node_name, node):
        """
        create plotly objects for node
        :param node_name: name of node
        :param node: node object
        :return: tuple with shape, marker and text objects
        """
        radius = self._get_node_radius(node)
        color = self.node_colors[node['type']]
        text_offset = 0.02 * self.xy_range
        
        shape = go.layout.Shape(type="circle",
                                xref="x", yref="y",
                                fillcolor=self.node_colors[node['type']],
                                x0=node['x'] - radius, y0=node['y'] - radius, 
                                x1=node['x'] + radius, y1=node['y'] + radius,
                                line=dict(color='white', width=1), 
                                layer='below')
        
        marker = go.Scatter(x=[node['x']], y=[node['y']],         
                mode='markers',
                hoverinfo='text',
                hovertext=self._get_node_hover_text(node_name, node),
                marker=dict(size=0, color=color),
                showlegend=False)
        
        text = go.layout.Annotation(x=node['x'], 
                                    y=node['y'] + radius + text_offset,
                                    xref='x',
                                    yref='y',
                                    text=node_name, 
                                    font=dict(color='white'),
                                    bgcolor='black',
                                    showarrow=False, 
                                    opacity=0.8)

        return shape, marker, text
    
    def _create_edge_object(self, edge, origin, destination):   
        """
        create plotly objects for edge
        :param edge: edge object
        :param origin: origin node object
        :param destination: destination node object
        :return: list with arrow and line objects
        """ 

        if 'flow' in edge and edge['flow'] > 0:
            line_type = 'solid'            
            color = self._get_edge_color(edge[self.edge_color_field]) if self.edge_color_field else 'gray'
        else:
            line_type = 'dash'
            color = 'gray'

        # get radius of origin and destination nodes
        origin_radius = self._get_node_radius(origin)
        destination_radius = self._get_node_radius(destination)

        # create vector from origin to edge of origin node along edge
        origin_radius_x, origin_radius_y = self._create_vector(origin['x'], origin['y'], destination['x'], 
                                                               destination['y'], origin_radius)
        destination_radius_x, destination_radius_y = self._create_vector(destination['x'], destination['y'], origin['x'], 
                                                                         origin['y'], destination_radius)
        # create vector from origin edge to destination edge
        origin_edge_x = origin['x'] + origin_radius_x
        origin_edge_y = origin['y'] + origin_radius_y
        destination_edge_x = destination['x'] + destination_radius_x
        destination_edge_y = destination['y'] + destination_radius_y

        width = self._get_edge_width(edge)

        arrow_trace = go.Scatter(        
            x=[origin_edge_x, destination_edge_x],
            y=[origin_edge_y, destination_edge_y],
            line=dict(width=width, color=color, dash=line_type),
            marker = dict(size=max(width*4, 5), symbol= "arrow-bar-up", angleref="previous", color=color),                            
            mode='lines+markers', 
            hoverinfo='skip', 
            showlegend=False)
        
        edge_trace = go.Scatter(
            x=np.linspace(origin_edge_x, destination_edge_x, 20),
            y=np.linspace(origin_edge_y, destination_edge_y, 20),

            line=dict(width=0, color=color, dash=line_type),
            hoverinfo='text',
            hovertext=self._get_edge_hover_text(edge),
            mode='lines', 
            showlegend=False)
                    
        return [arrow_trace, edge_trace]    

    def _create_colorbar_object(self):
        """
        create colorbar object
        :return: colorbar trace object
        """
        if not self.edge_color_field: return None
        colorbar_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=self.colormap_name,
                cmin=self.edge_df[self.edge_color_field].min(),
                cmax=self.edge_df[self.edge_color_field].max(),
                colorbar=dict(
                    title='Edge ' + self.edge_color_field.replace('_', ' ').title(),
                    titleside='right', 
                    titlefont=dict(color='white'),  # Set colorbar title font color to white
                    tickfont=dict(color='white'))    # Set colorbar tick font color to white        
            ),
            hoverinfo='none', 
            showlegend=False
        )
        return colorbar_trace
    
    def _create_legend_object(self):
        """
        create legend object
        :return: list with legend trace objects
        """
        legend_traces = []
        for name, color in self.node_colors.items():
            legend_trace = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color, line=dict(width=2)),
                name=name,
                showlegend=True
            )
            legend_traces.append(legend_trace)    
        return legend_traces
    