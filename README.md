# Prédiction du Churn Client

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-95%25-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Sommaire

- [Prédiction du Churn Client](#prédiction-du-churn-client)
  - [Sommaire](#sommaire)
  - [Objectif](#objectif)
  - [Gestion du projet](#gestion-du-projet)
  - [Technologies utilisées](#technologies-utilisées)
  - [Requirements](#requirements)
      - [Installer toutes les dépendances en une seule commande :](#installer-toutes-les-dépendances-en-une-seule-commande-)
  - [Installation et exécution](#installation-et-exécution)
  - [Visualisations](#visualisations)
  - [Tests unitaires](#tests-unitaires)

---

## Objectif

Ce projet a pour but de développer un **pipeline de Machine Learning** pour prédire le **churn client** dans une entreprise de télécommunications.

Objectifs principaux :

- Identifier les clients à risque de désabonnement.  
- Préparer et analyser les données (EDA) pour mieux comprendre les facteurs influençant le churn.  
- Entraîner et comparer plusieurs modèles supervisés : Logistic Regression et Random Forest.  
- Évaluer les modèles avec des métriques clés : Accuracy, Recall, F1-score, ROC-AUC.  

---

## Gestion du projet

Pour suivre une **approche professionnelle** :

- Tâches organisées avec **Jira**.  
- Branches GitHub pour les **notebooks d’EDA** et les **tests unitaires**.  
- Structure des fichiers **claire et modulaire**.  

<div align="center">
  <img src="images/structure_du_projet.png" alt="Structure du projet" width="45%">
</div>

---

## Technologies utilisées

- [Python](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [Pytest](https://docs.pytest.org/)  

---

## Requirements

Installer les bibliothèques nécessaires :

pandas
numpy
matplotlib
seaborn
scikit-learn
pytest

#### Installer toutes les dépendances en une seule commande :

pip install -r requirements.txt 


## Installation et exécution
1.Cloner le dépôt :
git clone https://github.com/ton-utilisateur/Churn-Prediction.git
cd Churn-Prediction

2.Installer les dépendances :
pip install -r requirements.txt

3.Lancer le projet :
Notebooks pour l’EDA : main/notebooks.ipynb
Pipeline Machine Learning : src/pipeline.py

4.Exécuter les tests unitaires :
pytest tests/


## Visualisations

--> ROC Curve
--> Distribution du churn

<div align="center"> 
    <img src="images/courbesROCComparatives.png" alt="ROC Curve" width="45%">

## Tests unitaires

Tous les tests sont situés dans le dossier tests/

Utilisation de pytest pour vérifier les fonctions du pipeline
Exemple de commande :

pytest tests/ -v