import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib 
import os
from typing import Tuple, Dict, List, Callable
from functools import partial

# ----------------------------------------------------------------------------
# Configuration initiale (constantes et configuration)
# ----------------------------------------------------------------------------
matplotlib.use('Agg')  # Mode non interactif pour sauvegarde

CONFIG = {
    "input_file": "simimat.csv",
    "separator": ";",
    "output_dir": "mst_output"
}

# ----------------------------------------------------------------------------
# Fonctions utilitaires pures
# ----------------------------------------------------------------------------
def ensure_output_dir_exists(output_dir: str) -> None:
    """Crée le répertoire de sortie s'il n'existe pas."""
    os.makedirs(output_dir, exist_ok=True)

def load_data(file_path: str, separator: str) -> Tuple[nx.Graph, List[str]]:
    """Charge les données depuis un fichier CSV et crée un graphe."""
    df = pd.read_csv(file_path, sep=separator, index_col=0)
    nodes = df.index.tolist()
    graph = nx.Graph()
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            weight = df.iloc[i, j]
            if pd.notna(weight):
                graph.add_edge(nodes[i], nodes[j], weight=weight)
    
    return graph, nodes

def validate_algorithm_choice(choice: str) -> bool:
    """Valide le choix de l'algorithme."""
    return choice.lower() in ['kruskal', 'prim', 'boruvka']

def validate_weight_choice(choice: str) -> bool:
    """Valide le choix du type de poids."""
    return choice.lower() in ['max', 'min']

def get_user_input(prompt: str, validator: Callable[[str], bool], error_msg: str) -> str:
    """Obtient une entrée utilisateur validée."""
    while True:
        choice = input(prompt).strip().lower()
        if validator(choice):
            return choice
        print(error_msg)

# ----------------------------------------------------------------------------
# Fonctions de visualisation
# ----------------------------------------------------------------------------
def calculate_optimal_k(G):
    n = len(G)
    if n == 0:
        return 1.0
    
    avg_degree = sum(dict(G.degree()).values()) / n
    # Formule hybride avec pondération
    k = 0.3 * np.log(n + 1) + 0.7 * (avg_degree ** 0.5)
    return np.clip(k, 0.5, 4.0)  # Borné entre 0.5 et 4


def save_graph(G: nx.Graph, filename: str, title: str) -> None:
    """Sauvegarde un graphe avec des arêtes proportionnelles aux valeurs."""
    # 1. Calcul des paramètres dynamiques
    n_nodes = len(G)
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / n_nodes if n_nodes > 0 else 0
    
    # 2. Calcul de la taille de la figure
    base_size = 12
    size_factor = np.log(n_nodes + 1) * 0.25 + avg_degree * 0.05
    figsize = (base_size + size_factor, base_size + size_factor * 0.8)
    
    # 3. Calcul du paramètre k optimisé
    k = calculate_optimal_k(G)
    
    plt.figure(figsize=figsize)
    
    # 4. Récupération des poids des arêtes (avec valeur par défaut si absent)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(edge_weights.values()) if edge_weights else 1
    min_weight = min(edge_weights.values()) if edge_weights else 1
    
    # Normalisation des poids entre 0.1 et 3.0
    edge_widths = []
    for edge in G.edges():
        weight = edge_weights.get(edge, 1)
        if max_weight != min_weight:
            normalized = 0.1 + 2.9 * (weight - min_weight) / (max_weight - min_weight)
        else:
            normalized = 1.0
        edge_widths.append(normalized)
    
    # 5. Layout avec paramètres optimisés
    pos = nx.spring_layout(
        G,
        k=k,
        iterations=1000,
        threshold=1e-5,
        seed=42,
        scale=1.0 + size_factor * 0.1,
        weight='weight'  # Prend en compte les poids pour le layout
    )
    
    # 6. Dessin des arêtes avec épaisseur proportionnelle (version corrigée)
    if n_nodes < 200:  # Pour les petits graphes: utiliser FancyArrowPatch
        nx.draw_networkx_edges(
            G, 
            pos, 
            alpha=0.25, 
            width=edge_widths,
            edge_color='#888888',
            arrows=True,  # Nécessaire pour les styles avancés
            arrowstyle='-',  # Style de flèche (trait simple)
            connectionstyle='arc3,rad=0.1'  # Courbure des arêtes
        )
    else:  # Pour les grands graphes: utiliser LineCollection (plus rapide)
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.25,
            width=edge_widths,
            edge_color='#888888',
            arrows=False  # Désactive les fonctionnalités avancées pour la performance
        )
        
    # 7. Dessin des labels avec taille adaptative
    for node, (x, y) in pos.items():
        degree = degrees[node]
        font_size = max(4, 10 * np.log10(degree + 3))
        
        plt.text(
            x, y,
            str(node),
            fontsize=font_size,
            ha='center',
            va='center',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='none',
                boxstyle='round,pad=0.2'
            ),
            zorder=4
        )
    
    # 8. Affichage des poids sur les arêtes (optionnel)
    if edge_weights and n_nodes < 50:  # Seulement si pas trop de nœuds
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_weights,
            font_size=8,
            label_pos=0.5,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.title(title, pad=20, fontsize=14 + size_factor * 0.5)
    plt.axis('off')
    plt.tight_layout(pad=1.0 + size_factor * 0.2)
    
    # 9. Sauvegarde avec résolution adaptative
    dpi = 300 if n_nodes < 100 else (200 if n_nodes < 500 else 150)
    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        transparent=False
    )
    plt.close()
    

    
# ----------------------------------------------------------------------------
#
#   Algorithmes MST (kruskal, prim, boruvka) 
#
# ----------------------------------------------------------------------------
def kruskal(graph: nx.Graph, weight_type: str = 'max') -> Tuple[nx.Graph, float]:
    """Implémentation fonctionnelle de l'algorithme de Kruskal."""
    edges = sorted(
        graph.edges(data=True), 
        key=lambda x: x[2]['weight'], 
        reverse=(weight_type == 'max')
    )
    
    parent = {node: node for node in graph.nodes()}
    mst = nx.Graph()
    mst_weight = 0.0

    def find(node: str) -> str:
        """Trouve la racine d'un nœud avec compression de chemin."""
        if parent[node] == node:
            return node
        parent[node] = find(parent[node])
        return parent[node]

    def union(u: str, v: str) -> bool:
        """Union de deux ensembles, retourne True si une union a été effectuée."""
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            return True
        return False

    for u, v, data in edges:
        if union(u, v):
            mst.add_edge(u, v, weight=data['weight'])
            mst_weight += data['weight']
            if mst.number_of_edges() == len(graph.nodes()) - 1:
                break
                
    return mst, mst_weight

def prim(graph: nx.Graph, weight_type: str = 'max') -> Tuple[nx.Graph, float]:
    """Implémentation fonctionnelle de l'algorithme de Prim."""
    start_node = next(iter(graph.nodes()))
    mst = nx.Graph()
    mst.add_node(start_node)
    mst_weight = 0.0
    
    available_edges = []
    comparator = (lambda x, y: x > y) if weight_type == 'max' else (lambda x, y: x < y)

    while mst.number_of_edges() < len(graph.nodes()) - 1:
        # Ajouter les nouvelles arêtes disponibles
        available_edges.extend(
            (u, v, data['weight'])
            for u, v, data in graph.edges(data=True)
            if (u in mst and v not in mst) or (v in mst and u not in mst)
        )
        
        if not available_edges:
            break
            
        # Trouver la meilleure arête selon le type de poids
        best_edge = max(available_edges, key=lambda x: x[2]) if weight_type == 'max' else min(available_edges, key=lambda x: x[2])
        available_edges.remove(best_edge)
        u, v, weight = best_edge
        
        # Déterminer quel nœud est nouveau
        new_node = v if v not in mst else u
        mst.add_edge(u, v, weight=weight)
        mst_weight += weight
        
        # Nettoyer les arêtes qui mènent à des nœuds déjà dans le MST
        available_edges = [
            edge for edge in available_edges 
            if edge[1] not in mst and edge[0] not in mst
        ]
    
    return mst, mst_weight

def boruvka(graph: nx.Graph, weight_type: str = 'max') -> Tuple[nx.Graph, float]:
    """Implémentation fonctionnelle de l'algorithme de Borůvka."""
    parent = {node: node for node in graph.nodes()}
    mst = nx.Graph()
    mst_weight = 0.0
    num_components = len(graph.nodes())
    
    def find(node: str) -> str:
        """Trouve la racine d'un nœud avec compression de chemin."""
        if parent[node] == node:
            return node
        parent[node] = find(parent[node])
        return parent[node]

    def union(u: str, v: str) -> bool:
        """Union de deux ensembles, retourne True si une union a été effectuée."""
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            return True
        return False

    while num_components > 1:
        cheapest = {}
        compare = (lambda x, y: x > y) if weight_type == 'max' else (lambda x, y: x < y)
        
        for u, v, data in graph.edges(data=True):
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                for root in [root_u, root_v]:
                    if root not in cheapest or compare(data['weight'], graph[cheapest[root][0]][cheapest[root][1]]['weight']):
                        cheapest[root] = (u, v)
        
        added_edges = set()
        for root in cheapest:
            u, v = cheapest[root]
            edge_key = tuple(sorted((u, v)))
            if edge_key not in added_edges and union(u, v):
                weight = graph[u][v]['weight']
                mst.add_edge(u, v, weight=weight)
                mst_weight += weight
                num_components -= 1
                added_edges.add(edge_key)
    
    return mst, mst_weight

# ----------------------------------------------------------------------------
# Fonctions de rapport et sauvegarde
# ----------------------------------------------------------------------------
def print_mst_results(mst: nx.Graph, weight: float, algorithm: str, weight_type: str) -> None:
    """Affiche les résultats du MST."""
    print(f"\nArbre Recouvrant {('Maximal' if weight_type == 'max' else 'Minimal')} trouvé par l'algorithme de {algorithm}:")
    print(f"Poids total de l'arbre : {weight:.3f}")
    print("Arêtes de l'arbre :")
    for u, v, data in mst.edges(data=True):
        print(f"{u} -- {v} : {data['weight']:.3f}")

def save_mst_visualization(mst: nx.Graph, output_dir: str, algorithm: str, weight_type: str) -> str:
    """Sauvegarde la visualisation du MST et retourne le chemin du fichier."""
    output_filename = os.path.join(output_dir, f"mst_{algorithm}_{weight_type}_labels.svg")
    title = f"Arbre Recouvrant {('Maximal' if weight_type == 'max' else 'Minimal')} ({algorithm.capitalize()}) - Labels Seuls"
    save_graph(mst, output_filename, title)
    return output_filename

# ----------------------------------------------------------------------------
# Programme principal (composé de fonctions pures)
# ----------------------------------------------------------------------------
def main() -> None:
    """Fonction principale du programme."""
    # Initialisation
    ensure_output_dir_exists(CONFIG["output_dir"])
    
    try:
        graph, nodes = load_data(CONFIG["input_file"], CONFIG["separator"])
        print(f"Fichier '{CONFIG['input_file']}' chargé avec succès. "
              f"Nombre de nœuds : {len(nodes)}, "
              f"Nombre d'arêtes : {graph.number_of_edges()}")
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{CONFIG['input_file']}' n'a pas été trouvé.")
        return

    # Sélection des algorithmes
    algorithm_choice = get_user_input(
        "Choisissez l'algorithme ('Kruskal', 'Prim', 'Boruvka'): ",
        validate_algorithm_choice,
        "Choix d'algorithme invalide. Veuillez choisir parmi 'Kruskal', 'Prim' ou 'Boruvka'."
    )
    
    weight_choice = get_user_input(
        "Choisissez le type d'arbre recouvrant ('max' pour maximal, 'min' pour minimal): ",
        validate_weight_choice,
        "Choix de type de poids invalide. Veuillez choisir 'max' ou 'min'."
    )

    # Mapping des algorithmes
    algorithms = {
        'kruskal': kruskal,
        'prim': prim,
        'boruvka': boruvka
    }
    
    # Exécution de l'algorithme sélectionné
    mst, mst_weight = algorithms[algorithm_choice](graph.copy(), weight_choice)
    
    # Affichage et sauvegarde des résultats
    if mst is not None:
        print_mst_results(mst, mst_weight, algorithm_choice.capitalize(), weight_choice)
        output_path = save_mst_visualization(
            mst, 
            CONFIG["output_dir"], 
            algorithm_choice, 
            weight_choice
        )
        print(f"\nLe graphe de l'arbre recouvrant (labels seuls) a été sauvegardé dans : {output_path}")
    else:
        print("Erreur lors de la construction de l'arbre recouvrant.")
    
    print(f"\nLes fichiers de sortie ont été générés dans : {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
