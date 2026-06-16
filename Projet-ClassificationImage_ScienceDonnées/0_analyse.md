---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "56952047c4310bf0c313ca44bb5af720", "grade": false, "grade_id": "cell-883bbb5e1919ca1e", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

# Analyse de données

:::{admonition} Consignes
:class: dropdown

Vous documenterez votre analyse de données dans cette feuille. Nous
vous fournissons seulement l'ossature. À vous de piocher dans les
feuilles et TPs précédents les éléments que vous souhaiterez
réutiliser et adapter pour composer votre propre analyse!

:::

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "07b6f7ec89dbe99e092902adb7ff9cc8", "grade": false, "grade_id": "cell-e15fdd0116a9b9e2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Import des bibliothèques

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "96489bd4f670f5daa152f95fa18ce301", "grade": false, "grade_id": "cell-406ad2a159064cfa", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On commence par importer les bibliothèques dont nous aurons besoin.

:::{admonition} Consignes
:class: dropdown

Inspirez vous des précédentes feuilles. N'oubliez pas d'importer
`utilities` et `intro_science_donnees`.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 8e6a7da13b6e1d41caa1c33a8c0104e5
  grade: false
  grade_id: cell-76e64ee7bcb96bb5
  locked: false
  schema_version: 3
  solution: true
  task: false
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e319f7fbbc65f7368fd94b546edb9e34", "grade": false, "grade_id": "cell-6ec03d05a4e1c693", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Chargement des images

:::{admonition} Consignes
:class: dropdown

- Chargez vos images recentrées et réduites.
- Affichez les.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: a842cb19c2f315d9a183cf8a17bd295f
  grade: false
  grade_id: cell-211e275ccc954714
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# Chargement de notre propre jeu de données : Teléphone allumé ( Plusieurs fond d'images) et éteint (Plusieurs télephones)
dataset_dir = 'data'
images = load_images(dataset_dir, "*.jpg")
# Affichage de toutes les images prises initialement
image_grid(images, titles=images.index)
```

```{code-cell} ipython3
# Dans l’ensemble, les images ne doivent pas dépasser 100x100 pixels d'après la consigne de la feuille

# On recadre et redimensionne les images en 95x95 pixels car sinon on voit pas dans notre cas 
# Voir modif dans utilities pour les paramètres de fonctions.
images_cropped = images.apply(crop_image) 
image_grid(images_cropped, titles=images.index)
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
# On cherche à detecter le centre de l'image
foreground_img = [my_foreground_filter(img)
            for img in images]
image_grid(foreground_img)
```

```{code-cell} ipython3
# On montre ce qu'on fait avec un exemple d'image : 
img = images.iloc[0]
foreground = my_foreground_filter(img)
coordinates = np.argwhere(foreground)
# On veut les coordonnées du barycentre des pixels de l’avant plan donc
# la moyenne des coordonnées sur les X et les Y -- afin d’estimer les coordonnées du centre du fruit. 
# On garde le résultat dans center
center = (np.mean(coordinates[:, 1]), np.mean(coordinates[:, 0]))
```

```{code-cell} ipython3
# Voici ce que cela nous donne en résumé en calculant les données du barycentre de l'objet: 
plt.scatter(coordinates[:,1], -coordinates[:,0], marker="x");
plt.scatter(center[0], -center[1], 300, c='r', marker='+',linewidth=5);
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
show_source(my_preprocessing)
```

```{code-cell} ipython3
# On applique le prétraitement à toutes les images et en le stockant dans clean_images
clean_images = images.apply(my_preprocessing)
image_grid(clean_images)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "90ece8faea4fbed29ee2f9101b82061d", "grade": false, "grade_id": "cell-e723e3fb5001bd97", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Prétraitement : extraction des attributs

:::{admonition} Consignes
:class: dropdown

- Choisissez entre des attributs ad-hoc, les votres ou une combinaison de certains attributs.
- N'oubliez pas de normaliser votre table une fois les traitements
  effectués.
  
:::

```{code-cell} ipython3
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
show_source(MatchedFilter)
# On reutilise le MatchFilter qu'on avait utilisé pour le projet avec le Smiley
```

```{code-cell} ipython3
show_source(brightness)
show_source(std_brightness)
# Voici deux des attributs que nous avons ajouté en amont ( avant qu'on sache le get_color)
```

```{code-cell} ipython3
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
# Epsilon : ne pas diviser par 0, et on n'oublie pas de standardiser le tab 
epsilon = sys.float_info.epsilon
df_class = df_features['class'] # On sauvegarde les valeurs des classes avant la normalisation pour le remplacer apreès
df_features = (df_features - df_features.mean())/(df_features.std() + epsilon)  
#Standardise les colonnes à l’exception de la colonne class pour   calculer les corrélations entre colonnes.

df_features.describe() # nouvelles statistiques de notre jeu de donnée pour verifier si c'est bien standardiser
```

```{code-cell} ipython3
#On ajoute nos étiquettes (1 pour la premiere classe, -1 pour l'autre) dans la dernière colonne :
df_features["class"] = df_class
df_features[df_features.isna()] = 0 # On remplace les Na par des 0 
df_features
```

```{code-cell} ipython3
# Et on affiche pour revérifier au cas ou !
df_features.describe()
# Décrire les valeurs des moyennes et de l'écart type et faire le lien avec la standardisation
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0b04fc60580d53165e039ab88751a746", "grade": false, "grade_id": "cell-a69bb602ef2221a4", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Sauvegarde intermédiaire

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "35a89b8dc927e0dd763e9cdf592c4bdf", "grade": false, "grade_id": "cell-8cb1846130b9cfe2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Consignes
:class: dropdown

Une fois que vous êtes satisfaits des attributs obtenus, faites en une
sauvegarde intermédiaire.

:::

```{code-cell} ipython3
os.makedirs('clean_data', exist_ok=True)
for name, img in clean_images.items():
    img.save(os.path.join('clean_data', os.path.splitext(name)[0]+".png"))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b11c92757f734a72bf65f47b9513ca5f", "grade": false, "grade_id": "cell-4576e2c2723b724d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Visualisation des données

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ba2beca9ed69fdecbf145787bb70f7af", "grade": false, "grade_id": "cell-2a67648ea9c07906", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Consignes
:class: dropdown

Visualisez les attributs et leur corrélation. Mettez en œuvre les
formes que vous jugerez les plus pertinentes (carte thermique, ...).

:::

```{code-cell} ipython3
df_features.describe(include="all")
```

```{code-cell} ipython3
plt.figure(figsize=(10, 6))
sns.heatmap(df_features, annot=True, cmap='coolwarm')  
plt.title("Heatmap des images de notre jeu de données selon les attributs")
plt.show()
```

```{code-cell} ipython3
corr = df_features.corr()
corr.style.format(precision=2).background_gradient(cmap='coolwarm')
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
# Sort by the absolute value of the correlation coefficient
sval = corr_large['class'][:-1].abs().sort_values(ascending=False)
ranked_columns = sval.index.values
print(ranked_columns) 

col_selected = ranked_columns[0:3]
df_features_final = pd.DataFrame.copy(df_features_large)
df_features_final = df_features_final[col_selected]

df_features_final['class'] = df_features_large["class"]
g = sns.pairplot(df_features_final, hue="class", markers=["o", "s"], diag_kind="hist")
```

Les matrices de corrélation montrent que plusieurs attributs sont très corrélés avec la classe :

- M (max RGB) a la corrélation la plus élevée
- (R+G+B)/3 et brightness sont aussi très corrélés positivement car ce sont les mêmes
- certains attributs comme greenness peuvent être négativement corrélés

- les images de téléphone allumé (classe 1) ont des valeurs plus élevées sur ces attributs
- les images de téléphone éteint (classe -1) ont des valeurs plus faibles

On comprend donc ici que la luminosité et l’intensité des pixels jouent un rôle important.
On voit aussi que :
M, (R+G+B)/3, brightness, R, G, B sont fortement corrélés entre eux
Il y a donc une redondance d’informations car ces attributs décrivent toutes entre autres la luminosité globale de l’image

Ici :

- un téléphone allumé => image lumineuse =>  RGB élevées => M élevé
- un téléphone éteint => image sombre =>  RGB faibles => M faible

M (max RGB) est l’attribut le plus discriminant, car il a la plus forte corrélation avec la classe et il capte directement la présence d’une forte intensité lumineuse

les attributs permettent clairement de séparer les deux classes entre allumé et éteint.
On peut supposer que 2 à 3 attributs (M, brightness, std_brightness) suffisent largement pour un modèle robuste car les autres sont pour la plupart liés aux 3 attributs qu'on veut choisir.

Pour les filtres allumés et éteins, ils ont une bonne corrélation mais ils semblent moins pertinent à prendre car ils utilisent déja des mesures de corrélation qui se basent sur les attributs principaux qu'on voulait choisir. Le MatchFilter se base sur la corrélation des templates et ici on voit bien que c'est la lumonisité qui prime.

En somme, les corrélations montrent que la luminosité est le facteur principal pour distinguer un téléphone allumé d’un téléphone éteint.
L’attribut M, qui est  l’intensité maximale des composantes RGB, semble être le plus discriminant.
On pourrait éventuellement réduire le nombre d’attributs  pour obtenir une bonne classification puisqu'il y en a beaucoup qui sont redondants.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b6afee00ce7826717b483c73488161ff", "grade": false, "grade_id": "cell-c103279002f68467", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Performance 

:::{admonition} Consignes
:class: dropdown

- Choisissez un ou plusieurs classificateurs et calculez leur
  performances.
- Comparez les performances selon les attributs utilisés.
- Si vous le souhaitez, vous pouvez aussi:
    - comparer un ou plusieurs classificateurs
	- comparer les performances selon le nombre de colonnes (pixels ou
      attributs) considérées.

:::

```{code-cell} ipython3
#Nous stockerons les performances d'un classificateur ici du plus proche voisin kNN dans une table de données 
performances = pd.DataFrame(columns = ['Traitement', 'perf_tr', 'std_tr', 'perf_te', 'std_te'])
performances
```

```{code-cell} ipython3
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
performances
```

```{code-cell} ipython3
#On vérifie les performances de notre classificateur sur ce large ensemble d’attributs et on les ajoute à notre table
performances.loc[1] = ["23 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

```{code-cell} ipython3
# Je remets la matrice de corrélation que j'avais mis ci-dessus : 
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
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_tr'], yerr=feat_lc_df['std_tr'], label='Training set')
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_te'], yerr=feat_lc_df['std_te'], label='Test set')
plt.xticks(np.arange(1, 22, 1)) 
plt.xlabel('Number of features')
plt.ylabel(sklearn_metric.__name__)
plt.legend(loc='lower right');
```

D'après la courbe d'apprentissage en fonction du nombre d'attributs, on constate que les performances de test augmentent jusqu'à les 4-7 premiers attributs (~0.95 de balanced accuracy) jusqu'à presque atteindre le maximum du pic test qui va sur la courbe de l'entraînement, puis se dégradent clairement au delà de 10 attributs (Surement à cause du surapprentissage : le modèle mémorise les données d'entraînement mais généralise mal). 

Cela correspond principalement aux attributs :  M, C, B,(R+G+B)/3 brightness, filtre_eteint et filtre_allumé.

Or, on avait dit au-dessus que : 
"M, (R+G+B)/3, brightness, R, G, B sont fortement corrélés entre eux
Il y a donc une redondance d’informations car ces attributs décrivent toutes entre autres la luminosité globale de l’image

les attributs permettent clairement de séparer les deux classes entre allumé et éteint.
On peut supposer que 2 à 3 attributs (M, brightness, std_brightness) suffisent largement pour un modèle robuste car les autres sont pour la plupart liés aux 3 attributs qu'on veut choisir.

Pour les filtres allumés et éteins, ils ont une bonne corrélation mais ils semblent moins pertinent à prendre car ils utilisent déja des mesures de corrélation qui se basent sur les attributs principaux qu'on voulait choisir. Le MatchFilter se base sur la corrélation des templates et ici on voit bien que c'est la lumonisité qui prime."

Donc, même avec ces perfomances il serait quand même plus pertinent et judicieux de prendre 3 attributs qui sont plutôt différents d'ou : M, brightness, std_brightness avec ces 2 derniers qui se complètent

```{code-cell} ipython3
# On exporte notreotre table df_features_final dans le fichier features_data.csv contenant les attributs utiles 
# On Sélectionne manuellement les  attributs les plus  pertinents et interessant
col_selected = ['M', 'brightness', 'std_brightness']
df_features_final = df_features_large[list(col_selected) + ['class']].copy()
# Exportation vers CSV
df_features_final.to_csv("features_data.csv")

df_features_final
```

```{code-cell} ipython3
# Validation croisée
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_final, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
performances.loc[2] = ["Les 3 attributs que nous avons choisi (analyse univariée)", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ae7f04d9aea4689c8f25f33fd8cddf30", "grade": true, "grade_id": "cell-c5a8e3d90bb35375", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

VOTRE RÉPONSE ICI
