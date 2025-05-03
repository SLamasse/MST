# Générateur d'Arbres Recouvrant de Poids Maximal/Minimal

Ce script Python permet de construire et de visualiser l'arbre recouvrant de poids maximal ou minimal (MST) d'un graphe non orienté pondéré, à partir d'un fichier CSV. Il implémente les algorithmes de Kruskal, Prim et Borůvka.
Il peut être utile dans le cadre de l'analyse de co-occurence

## Fonctionnalités

* **Lecture de données depuis un fichier CSV :** Le script lit une matrice de similarité (ou de distance) depuis un fichier CSV.
* **Implémentation de trois algorithmes MST :**
    * Kruskal
    * Prim
    * Borůvka
* **Choix du type d'arbre recouvrant :** L'utilisateur peut choisir de générer l'arbre recouvrant de poids maximal ou minimal.
* **Visualisation du graphe MST :** Le script génère une visualisation du MST avec des épaisseurs d'arêtes proportionnelles à leurs poids. Les labels des nœuds sont également affichés.
* **Sauvegarde de la visualisation :** La visualisation est sauvegardée au format SVG dans un répertoire de sortie spécifié.
* **Rapports :** Affiche le poids total du MST et la liste de ses arêtes.

## Prérequis

Assurez-vous d'avoir les bibliothèques Python suivantes installées :

```bash
pip install pandas numpy networkx matplotlib typing
```

## Il reste pas mal de choses à faire .... 
