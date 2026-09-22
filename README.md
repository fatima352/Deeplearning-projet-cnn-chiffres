# Deeplearning-projet-cnn-chiffres

Ne pas oublier d'activer l'environnement python et d'installer les dépendances nécessaires au bon fonctionnement des scripts. 

# Lancement de l'environnement : 

python -m venv venv
source venv/bin/activate


# Lancement de l'installation des dependances : 

python -m pip install -r requirements.txt

## Pour la partie augmentation, les fichiers sont trop lourds a déplacer. Je vous demande donc de lancer les scripts dans cet ordre là depuis le dossier scripts: 

python3 formater_mnist.py

python3 augmenter_donnees.py


## Pour la partie training

python3 training/training.py

L'entrainement sur les données EMNIST est sauvegardé dans modele_emnist_base.npz

L'entrainement sur les données personnelles augmentées est sauvegardé dans modele_chiffres_npz