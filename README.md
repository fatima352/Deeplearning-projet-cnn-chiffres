# Deeplearning-projet-cnn-chiffres

Ce projet a pour objectif de détecter et reconnaître des chiffres manuscrits à l'aide d'un réseau de neurones convolutif (CNN).

Avant d'exécuter les différents scripts du projet, il est nécessaire de créer un environnement virtuel Python et d'installer les dépendances.

## Installation

### 1. Créer et activer l'environnement virtuel

Depuis la racine du projet :

Sous Windows PowerShell :

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Sous Linux/macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

Vérifier que le bon interpréteur est actif :

```bash
python -c "import sys; print(sys.executable)"
```

## Préparation et augmentation des données

Les fichiers générés par l'augmentation des données étant trop volumineux pour être inclus dans le dépôt, ils doivent être générés localement.

Depuis la racine du projet, lancer les scripts dans l'ordre suivant :

```bash
python scripts/decoupage_grille.py
python3 scripts/formater_mnist.py
python3 scripts/augmenter_donnees.py
```

Ces scripts permettent de préparer les images puis de générer les données augmentées nécessaires à l'entraînement.

## Entraînement du modèle

Pour lancer l'entraînement :

```bash
python training/training.py
```

Deux modèles sont sauvegardés au cours du processus :

- `modele_emnist_base.npz` : modèle obtenu après l'entraînement sur les données EMNIST ;
- `modele_chiffres.npz` : modèle obtenu après l'entraînement sur les données personnelles augmentées.

## Lancement de l'interface Streamlit

L'interface permet d'importer une image contenant un ou plusieurs chiffres manuscrits.

Elle permet ensuite de :

- visualiser l'image importée ;
- détecter les chiffres présents dans l'image ;
- visualiser les chiffres extraits ;
- obtenir la prédiction du CNN pour chaque chiffre ;
- consulter le niveau de confiance associé à chaque prédiction ;
- afficher le détail des probabilités pour les chiffres de 0 à 9.

### 1. Activer l'environnement virtuel

Si l'environnement virtuel n'est pas déjà actif :

```powershell
.\.venv\Scripts\Activate.ps1
```

Sous Linux/macOS :

```bash
source .venv/bin/activate
```

### 2. Lancer l'application

Depuis la racine du projet :

```bash
streamlit run interface/app.py
```

L'application s'ouvre ensuite dans le navigateur.

## Vérifications

Depuis la racine du projet :

```bash
python -m compileall -q model backprop training evaluation interface scripts
python -m model.cnn
python -m backprop.gradient_check
python evaluation/evaluate.py
streamlit run interface/app.py
```

Le projet implémente actuellement un CNN NumPy, et non un Vision Transformer. La
détection de plusieurs chiffres repose sur OpenCV et des contours, puis le CNN
classe chaque chiffre extrait en 10 classes.