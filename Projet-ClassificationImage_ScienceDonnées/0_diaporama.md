---
jupytext:
  notebook_metadata_filter: rise
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  auto_select: first
  autolaunch: false
  backimage: fond.png
  centered: false
  controls: false
  enable_chalkboard: true
  height: 100%
  margin: 0
  maxScale: 1
  minScale: 1
  scroll: true
  slideNumber: true
  start_slideshow_at: selected
  transition: none
  width: 90%
---

+++ {"slideshow": {"slide_type": "slide"}, "editable": true, "tags": []}

# Titre

- Binôme: Lazim SAR, Adam El-Asrag
- Adresses mails: lazim.sar@universite-paris-saclay.fr adam.el-asrag@universite-paris-saclay.fr 
- [Dépôt GitLab](https://gitlab.dsi.universite-paris-saclay.fr/xxx.yyy/Semaine8/)

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

## Jeu de données
Description du jeu de données et motivation: en quoi est-ce intéressant ?

**Deux classes :**
- **Classe A - Téléphone allumé** (10 images) : le même téléphone photographié avec différents contenus affichés à l'écran (fond d'écran, applications, etc.)
- **Classe B — Téléphone éteint** (10 images) : 10 téléphones différents, écran noir

**Motivation :** La détection automatique de l'état d'un téléphone (allumé/éteint) peut servir dans des contextes de surveillance ,de consommation énergétique ou de contrôle parental

**Convention de nommage :** `a01.jpg`…`a10.jpg` pour les allumés, `b01.jpg`…`b10.jpg` pour les éteints.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
from scipy import signal
import pandas as pd
import numpy as np
import seaborn as sns
from glob import glob as ls
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric

from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# Chargement de notre propre jeu de données : Teléphone allumé ( Plusieurs fond d'images) et éteint (Plusieurs télephones)
dataset_dir = 'data'
images = load_images(dataset_dir, "*.jpg")
# Affichage de toutes les images prises initialement
image_grid(images, titles=images.index)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# Dans l’ensemble, les images ne doivent pas dépasser 100x100 pixels d'après la consigne de la feuille

# On recadre et redimensionne les images en 95x95 pixels car sinon on voit pas dans notre cas 
# Voir modif dans utilities pour les paramètres de fonctions.
images_cropped = images.apply(crop_image) 
image_grid(images_cropped, titles=images.index)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true, "tags": []}

## Prétraitement 

Décrivez les étapes de prétraitement effectuées : recadrage; réduction
de la résolution + lissage; choix d'attributs etc.  Voir la [feuille 2
sur une première analyse des données](2_premiere_analyse_ACP.md) et la
[feuille 3 sur l'extraction des
attributs](3_extraction_d_attributs.md)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# On implemente une fonction qui extrait le premier plan pour le séparer l'arrière plan 
def my_foreground_filter(img):
    """
    My foreground_filter, un filtre de premier plan adapté aux images de téléphones.
    
    Étapes :
    1. Applique foreground_redness_filter avec seuil theta=0.6
       (adapté pour détecter le coté sombre du téléphone)
    2. Inverse si le fond est clair (invert_if_light_background)
    3. Applique un filtre gaussien pour lisser le bruit
    4. Reconvertit en booléen (seuil > 0.5) car le filtre retourne des floats
       et transparent_background nécessite un tableau booléen pour M[foreground]
    """
    foreground = foreground_redness_filter(img, theta=0.6)
    foreground = invert_if_light_background(foreground)
    foreground = scipy.ndimage.gaussian_filter(foreground.astype(float), sigma=0.2)
    foreground = foreground > 0.5  # reconvertit en booléen
    return foreground
    
# On regroupe toutes les choses qu'on a faites au fil du temps en une unique fonction effectuant le prétraitement pour que ce soit plus facile
def my_preprocessing(img):
    """
    Prétraitement d'une image
    
    - Calcul de l'avant plan
    - Mise en transparence du fond
    - Calcul du centre
    - Recadrage autour du centre
    """
    foreground = my_foreground_filter(img)
    img = transparent_background(img, foreground)
    coordinates = np.argwhere(foreground)
    if len(coordinates) == 0: # Cas particulier: il n'y a aucun pixel dans l'avant plan
        width, height = img.size
        center = (width/2, height/2)
    else:
        center = (np.mean(coordinates[:, 1]), np.mean(coordinates[:, 0]))
    img = crop_around_center(img, center)
    return img

# Chargement des images prétraitées
clean_images = load_images('clean_data', "*.png")
image_grid(clean_images, titles=clean_images.index)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# On crée de la table d'attributs pour les téléphones allumés (a) et éteints (b).
# Les attributs de couleur (redness, greenness, blueness) devraient bien distinguer
# les téléphones : les allumés avec des écrans colorés, les éteints avec des écrans noirs.
# L'attribut  elongation devrait être similaire pour les deux classes car les téléphones ont tous une forme rectangulaire ).
# On a jouté un nouvel attribut: brightness (luminosité moyenne)  et son ecart type pour notre projet

#  Matched filters du Projet 1 (On le reutilise en bien )
# Dans le Projet 1, on avait utilisé MatchedFilter pour les smileys(filtre_sourire et filtre_triste). 
#On applique le même principe ici : un template moyen des téléphones allumés et un des éteints.
# Une image allumée devrait avoir une forte corrélation avec filtre_allume
# et faible avec filtre_eteint, et inversement.
images_allume = clean_images[clean_images.index.str.startswith('a')]
images_eteint = clean_images[clean_images.index.str.startswith('b')]
filtre_allume = MatchedFilter(images_allume)
filtre_eteint = MatchedFilter(images_eteint)

df_features = pd.DataFrame({
    # Attributs du Projet 1 réutilisés 
    
    'redness':        clean_images.apply(redness),
    'elongation':     clean_images.apply(elongation),
    # MatchedFilter : corrélation avec le template moyen de chaque classe
    'filtre_allume':  clean_images.apply(filtre_allume.match),
    'filtre_eteint':  clean_images.apply(filtre_eteint.match),
    #  attributs supplémentaires donnés au dessus de la question
    'greenness':      clean_images.apply(greenness),
    'blueness':       clean_images.apply(blueness),
    'perimeter':      clean_images.apply(perimeter),
    'surface':        clean_images.apply(surface),
    #  Nouveaux attributs pertinents pour notre projet (téléphone allumé/éteint) 
    # La luminosité moyenne illustre bien  directement écran allumé (clair) vs éteint (noir)
    'brightness':     clean_images.apply(brightness),
    # L'ecart type de luminosité est élevé si l'écran affiche des couleurs variées (allumé) et proche de 0 si l'écran est presque tout noir (éteint)
    'std_brightness': clean_images.apply(std_brightness),
    # Étiquettes: 1 = allumé (a), -1 = éteint (b)
    'class':          clean_images.index.map(lambda name: 1 if name[0] == 'a' else -1),
})

df_features
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# Epsilon : ne pas diviser par 0, et on n'oublie pas de standardiser le tab 
epsilon = sys.float_info.epsilon
df_class = df_features['class'] # On sauvegarde les valeurs des classes avant la normalisation pour le remplacer apreès
df_features = (df_features - df_features.mean())/(df_features.std() + epsilon)  
#Standardise les colonnes à l’exception de la colonne class pour   calculer les corrélations entre colonnes.

df_features.describe() # nouvelles statistiques de notre jeu de donnée pour verifier si c'est bien standardiser
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
#On ajoute nos étiquettes (1 pour la premiere classe, -1 pour l'autre) dans la dernière colonne :
df_features["class"] = df_class
df_features[df_features.isna()] = 0 # On remplace les Na par des 0 
df_features
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true, "tags": []}

## Classificateurs favoris et visualisation des données


Quel classificateur avez vous choisi ? Quels sont les résultats et les
taux d'erreur ? Montrez vos résultats! Après le cours en Semaine 8,
faites la [feuille 4 sur les classificateurs](4_classificateurs.md)

```{code-cell} ipython3
#Nous stockerons les performances d'un classificateur ici du plus proche voisin kNN dans une table de données 
performances = pd.DataFrame(columns = ['Traitement', 'perf_tr', 'std_tr', 'perf_te', 'std_te'])
from sklearn.neighbors import KNeighborsClassifier
# On importe une mesure de précision appelée balanced accuracy score, qui pondère la précision selon la taille du jeu de données
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
# On importe le  modèle KNN avec 3 voisins
sklearn_model = KNeighborsClassifier(n_neighbors=3)
```

```{code-cell} ipython3
# On prépare la modélisation des données d'entraînements et ceux des tests en utilisant la fonction df_cross_validate
show_source(df_cross_validate)
```

```{code-cell} ipython3
# Calcul des performances avec cross validate
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print(metric_name)
performances.loc[0] = ["6 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
#On vérifie les performances de notre classificateur sur ce large ensemble d’attributs et on les ajoute à notre table
performances.loc[1] = ["23 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

```{code-cell} ipython3
#Nous allons rajouter de nouveaux attributs sur les couleurs afin d’identifier s’il y a un attribut qui corrélerait davantage avec les étiquettes.
#On utilise la fonction get_colors()

df_color=clean_images.apply(get_colors)
#Voici les nouveaux attributs qu'on ajoute 
# 'R','G','B','M=maxRGB', 'm=minRGB', 'C=M-m', 'R-(G+B)/2',
# 'G-B', 'G-(R+B)/2', 'B-R', 'B-(G+R)/2', 'R-G', '(G-B)/C',
# '(B-R)/C', '(R-G)/C', '(R+G+B)/3', 'C/V' avec V = (R+G+B)/3.

# Première étape: on initialise df_features_large avec les attributs qu'on avait déjà calculé
df_features_large = df_features.drop("class", axis = 1)
# Deuxième étapeon ajoute les nouveaux attributs issus de get_colors()
df_features_large = pd.concat([df_features_large, clean_images.apply(get_colors)], axis=1)

epsilon = sys.float_info.epsilon # epsilon

df_features_large = (df_features_large - df_features_large.mean()) / (df_features_large.std() + epsilon)
df_features_large[df_features_large.isna()] = 0
df_features_large["class"] = clean_images.index.map(lambda name: 1 if name[0] == 'a' else -1) # D'après ce qu'on avait fait et donné au dessus
# Et on calcule sa matrice de corrélation : 
corr_large = df_features_large.corr()
corr_large.style.format(precision=2).background_gradient(cmap='coolwarm')
```

```{code-cell} ipython3
''' On cherche le nombre optimal d'attributs. Pour cela, on utilise la fonction `feature_learning_curve` qui trie 
les attributs par corrélation puis ajoute progressivement les features dans l’ordre. '''
show_source(feature_learning_curve)
```

```{code-cell} ipython3
feat_lc_df, ranked_columns = feature_learning_curve(df_features_large, sklearn_model, sklearn_metric)
```

```{code-cell} ipython3
# On exporte notreotre table df_features_final dans le fichier features_data.csv contenant les attributs utiles 
# On Sélectionne manuellement les  attributs les plus  pertinents et interessant
col_selected = ['M', 'brightness', 'std_brightness']
df_features_final = df_features_large[list(col_selected) + ['class']].copy()
# Exportation vers CSV
df_features_final.to_csv("features_data.csv")
# Validation croisée
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_final, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
performances.loc[2] = ["Les 3 attributs que nous avons choisi (analyse univariée)", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

## Interprétations
Pouvez vous commenter vos taux d'erreur ? Le jeu de données était il
trop simple ou trop compliqué ? Une photo en particulier était elle
biaisée ? Etc.

**Pourquoi ça marche bien ?**
- La différence de luminosité entre écran allumé et éteint est très forte donc les attributs R, G, B sont très discriminants
- Même avec seulement 3 attributs et 20 images, on atteint plus de 95% de balanced accuracy

**Images difficiles à classifier :**
- Téléphones allumés avec fond d'écran sombre donc ressemblent à des éteints
- Téléphones éteints avec reflet sur l'écran donc semblent avoir un peu de lumière

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

## Discussion et conclusion

Vous pourriez parler *par exemple* des biais potentiels de vos
données, de l'utilisation d'un tel projet dans la vraie vie, des
difficultés rencontrées et de comment vous les avez surmontées (ou
pas).

**Biais identifiés :**
- Les 10 téléphones allumés sont du même modèle donc le classificateur pourrait apprendre la forme spécifique de ce téléphone, pas l'état allumé/éteint en général
- Cependant les téléphones éteints sont de 10 modèles différents donc la taille et la forme du cadre varient
- Les fonds changent d'une photo à l'autre donc peut perturber l'extraction d'avant-plan

**Ce qu'on aurait pu améliorer :**
- Photographier plusieurs modèles différents dans les deux classes
- Inclure des téléphones allumés avec fond d'écran sombre pour complexifier la tâche
- Utiliser plus de 10 images par classe pour un meilleur apprentissage

**Applications réelles :**
- Détection automatique de téléphones allumés dans des lieux où c'est interdit (examens, réunions confidentielles)
- Surveillance de consommation d'énergie
- Systèmes de gestion de flotte d'appareils mobiles
