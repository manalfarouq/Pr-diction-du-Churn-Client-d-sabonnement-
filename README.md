# Prédiction du Churn Client

---

## Objectif

Ce projet a pour but de développer un **pipeline de Machine Learning** pour prédire le **churn client** dans une entreprise de télécommunications.

Les objectifs principaux sont :

- Identifier les clients à risque de désabonnement.  
- Préparer et analyser les données (EDA) pour mieux comprendre les facteurs influençant le churn.  
- Entraîner et comparer plusieurs modèles supervisés : Logistic Regression et Random Forest.  
- Évaluer les modèles avec des métriques clés : Accuracy, Recall, F1-score, ROC-AUC.  

---

## Gestion du projet

Pour suivre une **approche professionnelle** de gestion de projet :

- J’ai organisé mes tâches en utilisant **Jira**.  
- J’ai créé plusieurs **branches sur GitHub** pour sauvegarder mes changements, notamment pour les **notebooks d’EDA** et les **tests unitaires avec Pytest**.  
- La structure des fichiers a été pensée pour rester **claire et modulaire**.  

<div align="center">
  <img src="images/structure_du_projet.png" alt="Structure du projet" width="45%">
</div>

---

## Technologies utilisées

- [Python](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/) et [Seaborn](https://seaborn.pydata.org/) pour la visualisation  
- [Scikit-learn](https://scikit-learn.org/stable/) pour le Machine Learning  
- [Pytest](https://docs.pytest.org/) pour les tests unitaires  

---

## Requirements

Pour exécuter le projet, installez les bibliothèques nécessaires :

- [x] `pandas`  
- [x] `numpy`  
- [x] `matplotlib`  
- [x] `seaborn`  
- [x] `scikit-learn`  
- [x] `pytest`  

### Installer toutes les dépendances en une seule commande :

- [x] `pip install -r requirements.txt`

---

## Installation et exécution

1. Cloner le dépôt :  
   - [x] `git clone https://github.com/manalfarouq/Prediction-du-Churn-Client-desabonnement-.git`  
   - [x] `cd Churn-Prediction`

2. Installer les dépendances :  
   - [x] `pip install -r requirements.txt`

3. Lancer le projet :  
   - Notebooks pour l’EDA : `main/notebooks.ipynb`  
   - Pipeline Machine Learning : `src/pipeline.py`

4. Exécuter les tests unitaires :  
   - [ ] `pytest tests/ -v`

---

## Visualisations

- ROC Curve  
- Distribution du churn

<div align="center">
  <img src="images/courbesROCComparatives.png" alt="ROC Curve" width="45%">  
</div>
