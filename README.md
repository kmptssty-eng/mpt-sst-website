# trust_density_analytics.html(conceptual)

import networkx as nx
import pandas as pd

class TrustDensityAnalyzer:
    def __init__(self, interaction_data):
        """
        interaction_data: a DataFrame with columns ['source', 'target', 'weight', 'timestamp']
        weight: strength of interaction (e.g., amount, frequency)
        """
        self.data = interaction_data
        self.graph = None
        
    def build_graph(self, time_window=None):
        # If time_window is provided, filter data within that window
        if time_window:
            filtered_data = self.data[self.data['timestamp'] >= time_window]
        else:
            filtered_data = self.data
        
        # Create a directed graph (or undirected, depending on the nature of interactions)
        self.graph = nx.from_pandas_edgelist(filtered_data, 'source', 'target', ['weight'], create_using=nx.DiGraph())
        
    def compute_trust_metrics(self):
        if self.graph is None:
            self.build_graph()
            
        metrics = {}
        
        # 1. Density
        metrics['density'] = nx.density(self.graph)
        
        # 2. Average clustering coefficient (for undirected version)
        undirected_graph = self.graph.to_undirected()
        metrics['clustering'] = nx.average_clustering(undirected_graph)
        
        # 3. Reciprocity (for directed graphs)
        metrics['reciprocity'] = nx.reciprocity(self.graph)
        
        # 4. Centralization (degree centralization)
        degree_centrality = nx.degree_centrality(self.graph)
        metrics['degree_centralization'] = max(degree_centrality.values()) - sum(degree_centrality.values()) / len(degree_centrality)
        
        # 5. Transaction velocity (average weight per edge)
        if nx.get_edge_attributes(self.graph, 'weight'):
            weights = nx.get_edge_attributes(self.graph, 'weight').values()
            metrics['avg_transaction_velocity'] = sum(weights) / len(weights)
        else:
            metrics['avg_transaction_velocity'] = 0
        
        # 6. Trust Density Index (a composite metric - this is just an example)
        # We can define it as a weighted combination of the above metrics
        # For example: TDI = (density * 0.3 + clustering * 0.2 + reciprocity * 0.3 + (1 - degree_centralization) * 0.2) * avg_transaction_velocity
        metrics['trust_density_index'] = (
            metrics['density'] * 0.3 +
            metrics['clustering'] * 0.2 +
            metrics['reciprocity'] * 0.3 +
            (1 - metrics['degree_centralization']) * 0.2
        ) * metrics['avg_transaction_velocity']
        
        return metrics
