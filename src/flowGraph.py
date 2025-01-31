# FlowGraph class
from pydantic import BaseModel, model_validator
from typing import Optional
from networkx import DiGraph
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus
import time
import networkx as nx
import sys

from flowViz import FlowViz


# node and edge definitions
class Node(BaseModel, frozen=True):
    """
    class of network node with attributes:
    name: str - name of node
    demand: float - demand of node (if node is sink)
    supply: float - supply of node (if node is source)
    capacity: float - maximum flow out of node
    type: str - type of node
    x: float - x-coordinate of node
    y: float - y-coordinate of node
    variable_cost: float - cost per unit flow out of node
    fixed_cost: float - cost of selecting node
    """
    name: str
    demand: Optional[float] = 0.0
    supply: Optional[float] = 0.0
    capacity: Optional[float] = float('inf')
    type: Optional[str] = None
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    variable_cost: Optional[float] = 0.0
    fixed_cost: Optional[float] = 0.0

    @model_validator(mode='after')
    def validate(self):
        """
        validate if node definition are correct
        """
        # check that demand is non-negative
        if self.demand < 0 or self.demand == float('inf'): raise ValueError('demand must be non-negative and finite')
        # check that supply is non-negative
        if self.supply < 0: raise ValueError('supply must be non-negative')
        # check that capacity is non-negative
        if self.capacity < 0: raise ValueError('capacity must be non-negative')
        # check that variable_cost is non-negative
        if self.variable_cost < 0: raise ValueError('variable_cost must be non-negative')
        # check that fixed_cost is non-negative
        if self.fixed_cost < 0: raise ValueError('fixed_cost must be non-negative')
        return self
    

class Edge(BaseModel, frozen=True):
    """
    class of edge between two nodes with attributes:
    origin: 'Node' - origin node of edge
    destination: 'Node' - destination node of edge
    capacity: float - maximum flow through edge
    variable_cost: float - cost per unit flow through edge
    fixed_cost: float - cost of selecting edge
    """
    origin: Node
    destination: Node
    capacity: Optional[float] = float('inf')
    variable_cost: Optional[float] = 0.0
    fixed_cost: Optional[float] = 0.0
    
    @model_validator(mode='after')
    def validate(self):
        """
        validate of edge definition is correct
        """
        # check that node names are different
        if self.origin.name == self.destination.name: raise ValueError('origin and destination names must be different')
        # check that capacity is non-negative
        if self.capacity < 0: raise ValueError('capacity must be non-negative')
        # check that variable_cost is non-negative
        if self.variable_cost < 0: raise ValueError('variable_cost must be non-negative')
        # check that fixed_cost is non-negative
        if self.fixed_cost < 0: raise ValueError('fixed_cost must be non-negative')
        return self
        

class FlowGraph(DiGraph):
    """
    class to define and solve minimum cost flow problems
    """
    def __init__(self, incoming_graph_data=None, **attr):
        """
        initialize FlowGraph object
        :param incoming_graph_data: list of nodes and edges or FlowGraph object
        :param attr: additional attributes
        """
        self.soft_constr = False
        # check what input data is given
        if isinstance(incoming_graph_data, FlowGraph): 
            # get all nodes and edges
            graph_elements = incoming_graph_data.get_nodes() + incoming_graph_data.get_edges()
        elif isinstance(incoming_graph_data, list): 
            graph_elements = incoming_graph_data
        elif incoming_graph_data is None: 
            graph_elements = []
        else:
            raise ValueError('incoming_graph_data must be a list, FlowGraph object or None')
        
        # initialialize digraph
        super().__init__(None, **attr)
        # add nodes and edges
        for element in graph_elements:
            if isinstance(element, Node): self.add_node(element)
            elif isinstance(element, Edge): self.add_edge(element)
            else: raise ValueError('incoming_graph_data must be a list of Node elements or Edge elements')

        # initialize visualization object
        self.viz = FlowViz(self)

    def add_node(self, node):
        """
        add node to graph
        :param node: Node object
        """
        # check if node is a Node element
        if not isinstance(node, Node): raise ValueError('node must be a Node element')
        # add node to graph
        super().add_node(node.name, demand=node.demand, supply=node.supply, capacity=node.capacity, type=node.type, 
                         variable_cost=node.variable_cost, fixed_cost=node.fixed_cost, x=node.x, y=node.y)

    def add_nodes_from(self, nodes):
        """
        add multiple nodes to graph from list or dataframe
        :param nodes: list of Node elements or dataframe
        """
        # check if nodes are dataframe
        if isinstance(nodes, pd.DataFrame):
            # get columns to create nodes
            col_select = [col for col in nodes.columns if col in Node.model_fields.keys()]
            for _, node in nodes[col_select].iterrows():                
                self.add_node(Node(**node.to_dict()))
        elif nodes: 
            for node in nodes: self.add_node(node)

    def add_edge(self, edge):    
        """
        add edge to graph
        @param edge: Edge object
        """   
        # check if edge is an Edge element
        if not isinstance(edge, Edge): raise ValueError('edge must be an Edge element')
        # check if nodes exist
        if not edge.origin.name in super().nodes: self.add_node(edge.origin)
        if not edge.destination.name in super().nodes: self.add_node(edge.destination)

        # add edge to graph
        super().add_edge(edge.origin.name, edge.destination.name, capacity=edge.capacity, 
                         variable_cost=edge.variable_cost, fixed_cost=edge.fixed_cost)

    def add_edges_from(self, edges):
        """
        add multiple edges from dataframe or list of edges
        :param edges: list of Edge elements or dataframe
        """
        # check if edges are dataframe
        if isinstance(edges, pd.DataFrame):
            # get columns to create edges
            col_select = [col for col in edges.columns if col in Edge.model_fields.keys()]
            for _, edge in edges[col_select].iterrows():
                edge_dict = edge.to_dict()
                # get nodes
                origin = self.get_node(edge_dict.pop('origin'))
                destination = self.get_node(edge_dict.pop('destination'))
                self.add_edge(Edge(origin=origin, destination=destination, **edge_dict))
        elif edges: 
            for edge in edges: self.add_edge(edge)

    def add_weighted_edges_from(self, edges):   
        # override DiGraph method to add weighted edges     
        self.add_edges_from(edges)

    def get_node(self, name):
        """
        get node object by name
        """
        if name in super().nodes:
            data = super().nodes[name]
            return Node(name=name, demand=data['demand'], supply=data['supply'], capacity=data['capacity'], 
                                  type=data['type'], variable_cost=data['variable_cost'], fixed_cost=data['fixed_cost'], 
                                  x=data['x'], y=data['y'])
        else:
            raise ValueError(f'node {name} not found')
    
    def get_nodes(self):
        """
        get all node objects of graph
        """
        return [self.get_node(name) for name in super().nodes]
    
    def get_edge(self, origin_name, destination_name):
        """
        get edge of graph by origin and destination node names
        @param origin_name: name of origin node
        @param destination_name: name of destination node
        """
        if origin_name in super().nodes and destination_name in super().nodes:
            data = super().edges[origin_name, destination_name]
            return Edge(origin=self.get_node(origin_name), destination=self.get_node(destination_name), 
                                  capacity=data['capacity'], variable_cost=data['variable_cost'], 
                                  fixed_cost=data['fixed_cost'])
        else:
            raise ValueError(f'edge {origin_name}-{destination_name} not found')

    def get_edges(self):
        """
        get all edge objects of graph
        """
        return [self.get_edge(origin_name, destination_name) for origin_name, destination_name in super().edges]
    
    def _assign_decision_variables(self):
        """
        assign decision variables
        """
        # variable counts
        cont_count = bin_count = 0
        # assign decision variables to all edges
        for origin_name, destination_name, edge in super().edges.data():
            # create flow variable if capacity is positive or soft constraints are used
            edge['flow_var'] = (LpVariable(f"flow_{origin_name}-{destination_name}", lowBound=0, cat='Continuous') 
                                if edge['capacity'] > 0 or self.soft_constr else None)            
            if edge['flow_var'] is not None: cont_count += 1

            # check if capacity variable is needed
            if self.soft_constr and edge['capacity'] < float('inf'):
                edge['capacity_var'] = LpVariable(f"capacity_{origin_name}-{destination_name}", lowBound=0, 
                                                  cat='Continuous')
                cont_count += 1
            else:
                edge['capacity_var'] = None

            edge['selection_var'] = (LpVariable(f"selection_{origin_name}-{destination_name}", cat='Binary') 
                                     if edge['capacity'] > 0 and edge['fixed_cost'] > 0 else None)
            if edge['selection_var'] is not None: bin_count += 1

        # assign decision variables to all nodes
        for name, node in super().nodes.data():
            node['selection_var'] = LpVariable(f"selection_{name}", cat='Binary') if node['fixed_cost'] > 0 else None
            if node['selection_var'] is not None: bin_count += 1

            # check if supply variable is needed
            if self.soft_constr and node['supply'] > node['demand'] and node['supply'] < float('inf'):
                node['supply_var'] = LpVariable(f"supply_{name}", lowBound=0, cat='Continuous')
                cont_count += 1
            else:
                node['supply_var'] = None

            if node['capacity'] < float('inf') and self.soft_constr:
                node['capacity_var'] = LpVariable(f"capacity_{name}", lowBound=0, cat='Continuous')
                cont_count += 1
            else:
                node['capacity_var'] = None

        if self.verbose: print(f"Variable types: {cont_count} continuous, {bin_count} binary")

    def _assign_objective_function(self):
        """
        define objective function
        """
        objective = 0
 
        # add edge costs
        for _, _, edge in super().edges.data():
            if edge['selection_var'] is not None: objective += edge['selection_var'] * edge['fixed_cost']
            if edge['flow_var'] is not None: objective += edge['flow_var'] * edge['variable_cost']
            if edge['capacity_var'] is not None: objective += edge['capacity_var'] * self.soft_penalty
        
        # add node costs
        for node_name, node in super().nodes.data():
            # add node selection costs
            if node['selection_var'] is not None: objective += node['selection_var'] * node['fixed_cost']
            # add node variable costs
            if node['variable_cost'] > 0:
                for _, _, edge in super().out_edges(node_name, data=True):
                    objective += edge['flow_var'] * node['variable_cost']
            
            # add soft constraint costs
            if node['supply_var'] is not None: objective += node['supply_var'] * self.soft_penalty
            if node['capacity_var'] is not None: objective += node['capacity_var'] * self.soft_penalty

        self.prob += objective, 'Objective',

    def _assign_constraints(self):
        """
        define constraints
        """
        # count of contraints
        constr_count = 0
        # add capacity constraints for edges with fixed costs
        for origin_name, destination_name, edge in super().edges.data():
            # get capacity
            capacity = edge['capacity'] if edge['capacity'] < float('inf') else self.max_flow
            rhs = capacity
            if edge['selection_var'] is not None: rhs *= edge['selection_var']
            if edge['capacity_var'] is not None: rhs += edge['capacity_var']
            self.prob += edge['flow_var'] <= rhs, f"capacity_{origin_name}-{destination_name}",
            constr_count += 1
            
            # get origin node
            origin_node = super().nodes[origin_name]
            # check if origin node has a selection variable
            if origin_node['selection_var'] is not None:
                rhs = capacity * origin_node['selection_var'] 
                if edge['capacity_var'] is not None: rhs += edge['capacity_var']
                self.prob += (edge['flow_var'] <= rhs, f"node_selection_{origin_name}-{destination_name}",)
                constr_count += 1

        total_demand = total_supply = 0
        # add flow conservation constraints
        for node_name, node in super().nodes.data():
            # aggregate in and out flows
            in_flow = 0
            for _, _, edge in super().in_edges(node_name, data=True):
                if edge['flow_var'] is not None: in_flow += edge['flow_var']
            
            out_flow = 0
            for _, _, edge in super().out_edges(node_name, data=True):
                if edge['flow_var'] is not None: out_flow += edge['flow_var']

            # add node capacity contraint
            if node['capacity'] < float('inf'):
                self.prob += out_flow <= node['capacity'], f"node_capacity_{node_name}",
                constr_count += 1

            # check what type of node it is
            if node['demand'] == node['supply']:
                # transshipment node: in_flow = out_flow
                self.prob += in_flow == out_flow, f"flow_balance_{node_name}",
            else:
                # in_flow - out_flow >= demand - supply - supply_added
                rhs = node['demand'] - node['supply']
                if node['supply_var'] is not None: rhs -= node['supply_var']
                self.prob += in_flow - out_flow >= rhs, f"flow_balance_{node_name}",
            constr_count += 1

            # update total demand and supply
            total_demand += node['demand']
            total_supply += node['supply']

        if self.verbose:
            print(f"Constraints: {constr_count}")
            print(f"Total supply: {total_supply}, Total demand: {total_demand}")
 
    def _assign_variable_values(self, opt_found):
        """
        assign decision variable values if optimal solution found, otherwise set to None
        add utilisation, total fixed cost, total variable cost and total cost, inflow and outflow
        @param opt_found: bool - if optimal solution was found
        """
        # assign edge values        
        for _, _, edge in super().edges.data():
            # initialize values
            edge['flow'] = edge['utilisation'] = edge['total_variable_cost'] = None
            edge['selected'] = edge['total_fixed_cost'] = edge['total_cost'] = None
            edge['capacity_added'] = edge['capacity_penalty_cost'] = None
            # check if optimal solution found
            if opt_found and edge['flow_var'] is not None:                    
                edge['flow'] = edge['flow_var'].varValue                    
                edge['total_variable_cost'] = edge['flow'] * edge['variable_cost']
                edge['total_cost'] = edge['total_variable_cost']
                capacity = edge['capacity']

                # check if capacity variable is used
                if edge['capacity_var'] is not None: 
                    edge['capacity_added'] = edge['capacity_var'].varValue 
                    capacity = edge['capacity'] + edge['capacity_added']
                    edge['capacity_penalty_cost'] = edge['capacity_added'] * self.soft_penalty
                    edge['total_cost'] += edge['capacity_penalty_cost']                        
                    capacity += edge['capacity_added']

                # calculate utilisation
                edge['utilisation'] = (edge['flow'] / capacity if 0 < capacity < float('inf') else None)

                if edge['selection_var'] is not None: 
                    edge['selected'] = edge['selection_var'].varValue
                    edge['total_fixed_cost'] = edge['selected'] * edge['fixed_cost']
                    edge['total_cost'] += edge['total_fixed_cost']

        # assign node values
        for node_name, node in super().nodes.data():
            # initialize values
            node['inflow'] = node['outflow'] = node['total_variable_cost'] = node['total_cost'] = node['utilisation'] = None
            node['selected'] = node['total_fixed_cost'] = node['total_cost'] = None
            node['capacity_added'] = node['supply_added'] = node['capacity_penalty_cost'] = node['supply_penalty_cost'] = None
            if opt_found:                
                # calculate inflow and outflow
                node['inflow'] = sum(edge['flow'] for _, _, edge in super().in_edges(node_name, data=True) 
                                     if edge['flow'] is not None)
                node['outflow'] = sum(edge['flow'] for _, _, edge in super().out_edges(node_name, data=True) 
                                      if edge['flow'] is not None)
                capacity = node['capacity']
                node['total_variable_cost'] = node['outflow'] * node['variable_cost']
                node['total_cost'] = node['total_variable_cost']
                
                # check if node has selection variable
                if node['selection_var'] is not None: 
                    node['selected'] = node['selection_var'].varValue 
                    node['total_fixed_cost'] = node['selected'] * node['fixed_cost']
                    node['total_cost'] += node['total_fixed_cost']

                # check if node as capacity variable
                if node['capacity_var'] is not None:
                    node['capacity_added'] = node['capacity_var'].varValue
                    capacity += node['capacity_added']
                    node['capacity_penalty_cost'] = node['capacity_added'] * self.soft_penalty
                    node['total_cost'] += node['capacity_penalty_cost']

                node['utilisation'] = (node['outflow'] / capacity if 0 < capacity < float('inf') else None)
                
                # check if node has supply variable
                if node['supply_var'] is not None:
                    node['supply_added'] = node['supply_var'].varValue
                    node['supply_penalty_cost'] = node['supply_added'] * self.soft_penalty
                    node['total_cost'] += node['supply_penalty_cost']                    

    def _set_soft_penalty(self, soft_penalty=None):
        """
        assign penalty cost for violating constraints
        @param soft_penalty: float - penalty cost for violating constraints (default: total of all variable and fixed 
                             costs)
        """
        # set soft penalty
        if soft_penalty is not None:
            if soft_penalty <= 0: raise ValueError('soft_penalty must be positive')
            self.soft_penalty = soft_penalty
        else:
            # set soft penalty as total of all variable and fixed costs
            max_cost = 1
            max_cost += sum(edge['variable_cost'] + edge['fixed_cost'] for _, _, edge in super().edges.data())
            max_cost += sum(node['variable_cost'] + node['fixed_cost'] for _, node in super().nodes.data())
            self.soft_penalty = max_cost

    def min_cost_flow(self, soft_constr=False, soft_penalty=None, verbose=True):
        """
        run minimum cost flow optimization
        @param soft_constr: bool - use soft constraints (default: False)
        @param soft_penalty: float - penalty cost for violating constraints (default: total of all variable and fixed 
                             costs)
        @param verbose: bool - print optimization status (default: True)
        @return: status of optimization
        """
        self.verbose = verbose
        self.soft_constr = soft_constr
        if self.verbose: print(f"Soft constraints: {soft_constr}")

        # get sinks with no path from any source
        no_path_sinks = self._get_no_path_sinks()
        if no_path_sinks:
            if self.verbose: print(f"Sinks with no path from any source: {no_path_sinks}")
            return 'Infeasible'

        # check if soft penalty needs to be set
        if soft_constr: 
            self._set_soft_penalty(soft_penalty)
            if self.verbose: print(f"Soft penalty: {self.soft_penalty:.2f}")

        # get maximum flow
        self.max_flow = sum(node['demand'] for _, node in super().nodes.data() if node['demand'] > 0)

        start_time = time.time()
        # create LP problem
        self.prob = LpProblem("FlowGraph.min_cost_flow", LpMinimize)
        # assign decision variables
        self._assign_decision_variables()
        # assign objective function
        self._assign_objective_function()
        # assign constraints
        self._assign_constraints()
        if self.verbose: print(f"Model creation time: {time.time() - start_time:.2f} s")

        start_time = time.time()
        # solve LP problem
        self.prob.solve()
        solve_time = time.time() - start_time

        # get status
        status = LpStatus[self.prob.status]

        if status == 'Optimal':
            # get objective value
            objective = self.prob.objective.value()
            if self.verbose: print(f"Optimal solution found: {objective:.2f} in {solve_time:.2f} s")
        elif verbose:
            print(f"Optimization status: {status} in {solve_time:.2f} s")
        
        # assign variable values
        self._assign_variable_values(status=='Optimal')

        return status
    
    def _get_no_path_sinks(self):
        """
        get list of sinks with no path from any source
        @return: list of sink names
        """
        # get sinks and sources
        sources, sinks = [], []
        for node_name, node in super().nodes(data=True):
            if node['demand'] > node['supply']: sinks.append(node_name)
            elif node['supply'] > node['demand']: sources.append(node_name)

        # check that there is at least one path from source to sink
        no_path_sinks = []
        for sink in sinks:    
            if not any(nx.has_path(self, source, sink) for source in sources): no_path_sinks.append(sink)
        
        return no_path_sinks    
    
    def infeasibility_analysis(self, verbose=True):
        """
        analyse why the model is infeasible
        @param verbose: bool - print infeasibility analysis (default: True)
        @return: dictionary with infeasible nodes and edges
        """
        # optimize with soft constraints
        self.min_cost_flow(soft_constr=True, verbose=verbose)

        # get nodes that require additional capacity of supply
        node_df = self.nodes_to_df()
        col_select = ['name', 'type', 'demand', 'supply', 'capacity_added', 'supply_added']
        infeasible_node_df = node_df.query('capacity_added > 0 or supply_added > 0')[col_select]

        # get edges that require additional capacity
        edge_df = self.edges_to_df()
        col_select = ['origin', 'destination', 'capacity', 'capacity_added']
        infeasible_edge_df = edge_df.query('capacity_added > 0')[col_select]


        if verbose and (not infeasible_node_df.empty or not infeasible_edge_df.empty):
            print("\nInfeasibility analysis:")

            for _, node in infeasible_node_df.iterrows():
                if node['capacity_added'] and node['capacity_added'] > 0: 
                    print(f"Node {node['name']} requires additional capacity: {node['capacity_added']}")
                if node['supply_added'] and node['supply_added'] > 0: 
                    print(f"Node {node['name']} requires additional supply: {node['supply_added']}")

            for _, edge in infeasible_edge_df.iterrows():
                if edge['capacity_added'] and edge['capacity_added'] > 0:
                    print(f"Edge {edge['origin']}-{edge['destination']} requires additional capacity: {edge['capacity_added']}")

        return {'nodes': infeasible_node_df, 'edges': infeasible_edge_df}
    
    def nodes_to_df(self):
        """
        convert all nodes to dataframe
        @return: dataframe with node attributes
        """
        # columns to be dropped
        drop_cols = ['selection_var', 'supply_var', 'capacity_var']
        # convert nodes to dataframe
        if not self.soft_constr: drop_cols += ['supply_added', 'capacity_added', 'supply_penalty_cost', 
                                               'capacity_penalty_cost']
        df = (pd.DataFrame(dict(super().nodes(data=True))).T
              .reset_index()
              .rename(columns={'index': 'name'})
              .drop(columns=drop_cols, errors='ignore'))
        return df
    
    def edges_to_df(self):
        """
        convert all edges to dataframe
        @return: dataframe with edge attributes
        """
        # columns to be dropped
        drop_cols = ['flow_var', 'selection_var', 'capacity_var']
        if not self.soft_constr: drop_cols += ['capacity_added', 'capacity_penalty_cost']
        # convert edges to dataframe
        df = (pd.DataFrame([{**{'origin': o, 'destination': d}, **attr} for o, d, attr in super().edges(data=True)])
              .drop(columns=drop_cols, errors='ignore'))
        return df

