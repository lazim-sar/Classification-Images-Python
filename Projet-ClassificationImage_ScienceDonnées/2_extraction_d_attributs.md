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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4e0e153a44dc5f4793169f37ee7486b8", "grade": false, "grade_id": "cell-cd0cec4179e6c7e1", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

# Extraction d'attributs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "30e91a629323925df1b39131af22aa98", "grade": false, "grade_id": "cell-363b8f6e2af016ab", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Les traitements réalisés dans cette feuille sont majoritairement des
rappels des semaines précédentes. **Le but de cette feuille est de
vous fournir une trame afin d'appliquer sereinement les traitements
vus les semaines précédentes**

La feuille comprends trois parties:
1. Le prétraitement de vos données par extraction d'avant plan,
   recentrage et recadrage (rappels de la semaine 6)
2. L'extraction d'attributs ad-hoc (rappels de la semaine 4)
3. La sélection d'attributs pertinents

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0ee9aa2d8b403bae6b39a1ceb6baeca3", "grade": false, "grade_id": "cell-c7ef38442d032da9", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vous exécuterez cette feuille  avec votre
propre jeu de données, en visualisant les résultats à chaque étape, et
en ajustant comme nécessaire.

```{code-cell} ipython3
"""Toutes les fonctions utilisées mais non définies dans ce notebook
(`crop_image`, `redness`, etc.)
sont définies dans le fichier `utilities.py.
N'hésitez pas à consulter son contenu avec `show_source(...)."""
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7a99d311f2ca1bda4c60c8ed01ff6176", "grade": false, "grade_id": "cell-76a04ac85061d8e0", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## 1. Prétraitement (rappels de la semaine 6)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "40f81c37a3f766dbad9682a8121609f5", "grade": false, "grade_id": "cell-329ade064e61886f", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous allons détecter le centre de l'objet et recadrer l'image sur ce
centre (comme en Semaine 6). Nous ferons ensuite une sauvegarde
intermédiaire: Enfin nous enregistrerons les images prétraitées dans
un dossier `clean_data`.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "38c730e7bf0343fd9b430ff9e09abe89", "grade": false, "grade_id": "cell-08fdab925024f2c8", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Import des bibliothèques

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0726b06d5f3e8b6e37f2f5e806d8f90e", "grade": false, "grade_id": "cell-4645b8bbd98453be", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous commençons par importer les bibliothèques dont nous aurons
besoin. Comme d'habitude, nous vous fournissons quelques utilitaires
dans le fichier `utilities.py`. Vous pouvez ajouter vos propres
fonctions au fur et à mesure du projet:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b65a8122f3f79731339646783699b0c4
  grade: false
  grade_id: cell-fbc4ace0d3351477
  locked: true
  schema_version: 3
  solution: false
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e7a0d62653b7330e42cd30b4629af313", "grade": false, "grade_id": "cell-0b51395eaa6045fa", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Import des données

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3850541009c0953b26f14f45a8ba5c91", "grade": false, "grade_id": "cell-1dfc81a33d46608e", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Chargez vos données dans images. Si vous n'en avez pas encore, chargez le jeu de données 'ApplesAndBananas' en décommentant le code ci-dessous et en commentant les 2 premières lignes de code.
:::

```{code-cell} ipython3
# Chargement de votre propre jeu de données
dataset_dir = 'data'
images = load_images(dataset_dir, "*.jpg")

# Chargement du jeu de données ApplesAndBananas
# dataset_dir = os.path.join(data.dir, 'ApplesAndBananas')
# images = load_images(dataset_dir, "?[012]?.png")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "631da584f219cdbb41f60a00471a0cd3", "grade": false, "grade_id": "cell-5adc847bb68d66fe", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous avons supposé que les images sont au format png. Si elles sont au
format jpeg il faut modifier les arguments de la fonction load_images(). 

*Remarque:* Nous allons aborder des jeux de données **plus grands** que
les précédents, en nombre et en résolution des images; il faut donc
être **un peu prudent** et parfois **patient**. Par exemple, le jeu de
données des pommes et bananes contient plusieurs centaines
d'images. Cela prendrait du temps pour les afficher et les traiter
toutes dans cette feuille Jupyter. Aussi, ci-dessus, utilisons nous le
glob `?[012]?.png` pour ne charger que les images dont le nom fait
moins de trois caractères et dont le deuxième caractère est 0, 1, ou 2.


Dans les exemples suivants, nous n'afficherons que les
premières et dernières images :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 2e6a7bd07d7f49445099723ba3e59859
  grade: false
  grade_id: cell-d923f69700a99d5a
  locked: true
  schema_version: 3
  solution: false
  task: false
---
head = images.head()
image_grid(head, titles=head.index)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b9cc1f13c945e385fd83fece2cda7eed
  grade: false
  grade_id: cell-8cf34d2bfb08d883
  locked: true
  schema_version: 3
  solution: false
  task: false
---
tail = images.tail()
image_grid(tail, titles=tail.index)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "109082c64a3fdecc9894d7808d649114", "grade": false, "grade_id": "cell-f5bb595ef24c5378", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Redimensionner et recadrer

Comme vu au TP6, il est généralement nécessaire de redimensionner et/ou de recadrer les
images. Lors du premier projet, nous avons utilisé des images
recadrées pour se simplifier la tâche. Si l'objet d'intérêt est petit
au milieu de l'image, il est préférable d'éliminer une partie de
l'arrière-plan pour faciliter la suite de l'analyse. De même, si
l'image a une résolution élevée, on peut la réduire pour accélérer les
calculs sans pour autant détériorer les performances. Dans l'ensemble,
les images ne doivent pas dépasser 100x100 pixels.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "31e5c46d762a8c47e3adcbb152eeabae", "grade": false, "grade_id": "cell-0e696cca3d087f2d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans les cellules suivantes, on recadre et redimensionne les images en
32x32 pixels. Aucun effort particulier n'est fait pour centrer l'image
sur le fruit. N'hésitez pas à améliorer cette étape comme en Semaine6.


:::{admonition} Exercice

Dans utilities, finissez la fonction crop_image en appliquant crop à l'image.
:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: e2374831206a0cfdac5080650a942384
  grade: false
  grade_id: cell-7cdc46f6970261cb
  locked: true
  schema_version: 3
  solution: false
  task: false
---
images_cropped = images.apply(crop_image)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: d98474df404102702ea43153bef9a658
  grade: false
  grade_id: cell-78735a6fe92120ab
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid(images_cropped.head())
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e3926fde21a9337c461af615be68d081", "grade": false, "grade_id": "cell-71f7a30f09a3597f", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Extraction de l'avant-plan

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a542e8e21c004961c0aca4e856dc72ae", "grade": false, "grade_id": "cell-ed4a0600c5d07d01", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Pour trouver le centre de l'objet, il faut déjà arriver à séparer les
pixels qui appartiennent à celui-ci de ceux qui appartiennent au
fond. Souvenez-vous, vous avez déjà fait cela en Semaine 6 avec la
fonction `foreground_filter`.

Dans l'exemple suivant, on utilise une variante
`foreground_redness_filter` qui sépare l'objet du fond en prenant en
compte la rougeur de l'objet et pas uniquement les valeurs de
luminosité.


:::{attention}

Comme pour `foreground_filter`, cette fonction a besoin d'un seuil
(compris entre 0 et 1) sur les valeurs des pixels à partir duquel on
décide s'il s'agit de l'objet ou du fond. Nous avons défini cette fonction pour qu'elle s'applique à notre jeu de données de pommes et de bananes. Avec les pommes et les
bananes, on se contentera de la valeur 2/3. Avec vos données, faites
varier cette valeur de seuil pour trouver la valeur qui semble la plus
adéquate.

:::

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b394f6ee81cbb27138ac999ffc739826", "grade": false, "grade_id": "cell-b9592dfe4d1b1c51", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Regardez foreground_redness_filter et modifiez le seuil pour l'appliquer à votre jeu de données.

:::

```{code-cell} ipython3
foreground_imgs = [foreground_redness_filter(img, theta=.5)
            for img in images]

image_grid(foreground_imgs)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ee4dd2acd95084dc1fa40a42ca9ac228", "grade": false, "grade_id": "cell-0ec125a40745d032", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On peut voir que selon s'il s'agit d'objets sombres sur fond clair ou
d'objets clairs sur fond sombre, on n'obtient pas les mêmes valeurs
booléenne en sortie de `foreground_filter`. La fonction
`invert_if_light_background` inverse simplement les valeurs booléennes
si une majorité de `True` est détectée. Voilà le résultat.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: fd93f157bac99a49b05f0d27d73d8af1
  grade: false
  grade_id: cell-68d213f1b8dde05d
  locked: true
  schema_version: 3
  solution: false
  task: false
---
foreground_imgs = [invert_if_light_background(foreground_redness_filter(img, theta=.75))
            for img in images]
image_grid(foreground_imgs)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8daa19401c5e9ab20a79ca5dea758a2b", "grade": false, "grade_id": "cell-1a0bd73b2eb64491", "locked": true, "schema_version": 3, "solution": false, "task": false}}

C'est légèrement mieux mais les images restent très bruitées; on
choisit d'appliquer un filtre afin de réduire les pixels isolés. C'est
ce qu'on fait avec la fonction `scipy.ndimage.gaussian_filter()`.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 3df6dbd2ef4e88d23330037162d0aade
  grade: false
  grade_id: cell-a7e311f99c04d100
  locked: true
  schema_version: 3
  solution: false
  task: false
---
foreground_imgs = [scipy.ndimage.gaussian_filter(
              invert_if_light_background(
                  foreground_redness_filter(img, theta=.6)),
              sigma=.2)
            for img in images]
image_grid(foreground_imgs)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ef32402d850130748c6dec1481c157da", "grade": false, "grade_id": "cell-edcd1412e8780f7d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Les traitements précédents résultent d'un certain nombre de choix
faits au fil de l'eau (seuil, filtre par rougeur, débruitage).

:::{admonition} Exercice

Définissez ci-dessous la fonction `my_foreground_filter` qui prend en
entrée une image et:
1. lui applique `foreground_redness_filter` **ou bien votre propre
   fonction** qui permet de filter votre objet.
3. inverse les valeurs booléennes si le fond est blanc avec la
   fonction `invert_if_light_background`
4. applique une convolution avec un filtre gaussien `scipy.ndimage.gaussian_filter` et supprime les pixels isolés avec un
   sigma à 0.2 (ou une autre valeur selon votre jeu de données).

Cette fonction extrait donc l'objet de l'arrière-plan

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: b50d25ea788ea72cd35df29e44ae181f
  grade: false
  grade_id: cell-388f0e8b70d29570
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "376de7ab06ef71f47d4575cf5a9d639f", "grade": false, "grade_id": "cell-ab925e70b4d327ad", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Cette fonction fait partie intégrale de la **narration des
traitements** que nous menons dans cette feuille. C'est pour cela que
nous la définissons directement dans cette feuille, et non dans
`utilities.py` comme on l'aurait fait pour du code réutilisable.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d6f1c55d9e1b3a3d4e6e04ce6bd4a462", "grade": false, "grade_id": "cell-3f74f24b54525ea2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Détection du centre de l'objet

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "352309533c089372fb3a97cbfde1ac79", "grade": false, "grade_id": "cell-7ccd60ee0acbc858", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous allons à présent déterminer la position du centre de l'objet, en
prenant la moyenne des coordonnées des pixels de l'avant plan.

Faisons cela sur la première image:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 35b9952ce12e44f194a247022ecc3fa9
  grade: false
  grade_id: cell-688d32bef4b64ef2
  locked: true
  schema_version: 3
  solution: false
  task: false
---
img = images.iloc[0]
img
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a3339f8f7ede35fa4ca828e8c677cb19", "grade": false, "grade_id": "cell-71d4072d46cfa6cd", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Calculez l'avant plan de l'image dans`foreground` avec la fonction
`my_foreground_filter` que vous venez de créer. 

On extrait ensuite les coordonnées de ses pixels :

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 33a7087886e2222cdcf0ea8d1aef10f7
  grade: false
  grade_id: cell-456706f2053e3a68
  locked: false
  schema_version: 3
  solution: true
  task: false
---
foreground = my_foreground_filter(img)
coordinates = np.argwhere(foreground)
coordinates
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9dd442cfc7030c0eb8de05ba860e15dc", "grade": false, "grade_id": "cell-2fafad09bd25eea7", "locked": true, "schema_version": 3, "solution": false, "task": false}}

que l'on affiche comme un nuage de points. Notez que la fonction `np.argwhere` renvoie (ligne, colonne). De plus, les coordonnées d’image et les coordonnées mathématiques n’utilisent pas la même convention (le 0 de l'axe vertical est respectivement en bas ou en haut). Voilà pourquoi dans le scatter plot on met coordinate[:,1] et -coordinates [:,0]:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: c84f889586fad16d7168e0a777f06cf2
  grade: false
  grade_id: cell-91d0510e1a875614
  locked: true
  schema_version: 3
  solution: false
  task: false
---
plt.scatter(coordinates[:,1], -coordinates[:,0], marker="x");
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0a052f8b4a95c0667a26f30a8fc3cc90", "grade": false, "grade_id": "cell-1f208b3395f73b47", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Calculez les coordonnées du barycentre des pixels de l'avant plan --
c'est-à-dire la moyenne des coordonnées sur les X et les Y -- afin
d'estimer les coordonnées du centre du fruit. On garde le résultat
dans `center`. Pour rappel, vous avez déjà calculé cela en Semaine 6.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 109c9e40583d4102208a8f672b4bfaa9
  grade: false
  grade_id: cell-1b73566df4dc76d3
  locked: false
  schema_version: 3
  solution: true
  task: false
---
center = (np.mean(coordinates[:, 1]), np.mean(coordinates[:, 0]))
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b54b887bb8f8e33cc10ca3b9614556da
  grade: false
  grade_id: cell-c4d67ad00158fe1e
  locked: true
  schema_version: 3
  solution: false
  task: false
---
plt.scatter(coordinates[:,1], -coordinates[:,0], marker="x");
plt.scatter(center[0], -center[1], 300, c='r', marker='+',linewidth=5);
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d3e6ecf43510fa4fe9b05a50777ac007", "grade": false, "grade_id": "cell-f279a67a6adb0a13", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Ce n'est pas forcément parfait: du fait des potentiels pixels isolés mais qui ont
été détectées comme de l'avant plan, le centre calculé n'est pas forcément à l'endroit que souhaité. Mais cela reste un bon début.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "675bd5d08aab92812118e642faee66eb", "grade": false, "grade_id": "cell-b44c46cf2c257a9b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Recadrage

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "49c0bea8053a3fe0b67b59cd0502c9d2", "grade": false, "grade_id": "cell-73b745f379708a66", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Maintenant que nous avons (approximativement) détecté le centre de l'objet, il nous suffit de recadrer autour de ce centre. Une fonction
`crop_around_center` est fournie pour cela. Comparons le résultat de
cette fonction par rapport à `crop_image` utilisé en [feuille2](2_premiere_analyse_ACP.md) :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: a1561781c018d076418a35ada53b637d
  grade: false
  grade_id: cell-4044e1d8ca3d20b6
  locked: true
  schema_version: 3
  solution: false
  task: false
---
crop_image(img) 
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7dc20f7ad4daa7ed8a8c0da9072c7a3b", "grade": false, "grade_id": "cell-6a73f050a21dd317", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Finissez d'écrire la fonction `crop_around_center`. Il ne manque que les coordonnées à droite (right) et en bas (bottom).

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: eebc19814314dd4123bdcc38e96aeb9c
  grade: false
  grade_id: cell-5ad18b7f58bfacc9
  locked: true
  schema_version: 3
  solution: false
  task: false
---
crop_around_center(img, center)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "72142c7da0bbb12ce0c416a65d6b8221", "grade": false, "grade_id": "cell-3e34f81c400abb8c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On constate que le recadrage sur le fruit est amélioré, même si pas
encore parfait.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "594673c7c1fee6a4d04f77e4eb8fd414", "grade": false, "grade_id": "cell-d56b644d2b458540", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Récapitulatif du prétraitement

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fb26eaea60feac92b40943ac8e8ee3b8", "grade": false, "grade_id": "cell-1c80da3d9ef180d1", "locked": true, "schema_version": 3, "solution": false, "task": false}}

À nouveau, nous centralisons tous les choix faits au fil de l'eau en
une unique fonction effectuant le prétraitement<a
name="choix_pretraitement"></a>. Cela facilite l'application de ce
traitement à toute image et permet de documenter les choix faits :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: f4c1d3cf6fa19b9b2538571fbfd83e53
  grade: false
  grade_id: cell-a031ad0180c19815
  locked: true
  schema_version: 3
  solution: false
  task: false
---
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
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: d47edb6af8253419dfb0e2665627cd48
  grade: false
  grade_id: cell-9073d5973279ee91
  locked: true
  schema_version: 3
  solution: false
  task: false
---
plt.imshow(my_preprocessing(images.iloc[0]));
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ea788113ceeffc653bb0995fc9e6ff23", "grade": false, "grade_id": "cell-684d382ecf73fbd4", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Appliquez le prétraitement à toutes les images que vous mettrez dans `clean_images`.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 16e2c4434177cac64f4f1c29e8c0f504
  grade: false
  grade_id: cell-e6ba50c596ac4cb7
  locked: false
  schema_version: 3
  solution: true
  task: false
---
clean_images = images.apply(my_preprocessing)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: c8756c133dcc20fd0733b97b05db8be8
  grade: false
  grade_id: cell-072b37a051ad21b8
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid(clean_images)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e6822c094fe0641ed5bce87a5ea3b607", "grade": false, "grade_id": "cell-f7d03b3fc781b97a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Sauvegarde intermédiaire

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "dabae3f947381b36b45a76d17421f00c", "grade": false, "grade_id": "cell-db2821b8e4311df0", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous sauvegardons maintenant les images prétraitées dans le répertoire
`clean_data` au format `PNG` :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 2dcc9e7946e5448680eaeb42a733984a
  grade: false
  grade_id: cell-bff603f0c67536ee
  locked: true
  schema_version: 3
  solution: false
  task: false
---
os.makedirs('clean_data', exist_ok=True)
for name, img in clean_images.items():
    img.save(os.path.join('clean_data', os.path.splitext(name)[0]+".png"))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7373d179996db56f34c3c21c2d3da840", "grade": false, "grade_id": "cell-a3a9fe4bdfe6d657", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Explication
:class: hint

`splitext` sépare un nom de fichier de son extension :

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: a475d6bc3b31a429a5324fc4622e36cd
  grade: false
  grade_id: cell-6f60c622fcabfa42
  locked: true
  schema_version: 3
  solution: false
  task: false
---
os.path.splitext("machin.jpeg")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "009c6b03b651da32e6ff30a503d4d11f", "grade": false, "grade_id": "cell-d241acc2b2ed27f7", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Extraction des attributs (rappels de la semaine 4)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ac1d945c894e74ab94f41166a0b9056d", "grade": false, "grade_id": "cell-3114446f2f5ae708", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Durant les semaines précédentes, vous avez déjà implémenté des
attributs tels que :
- La rougeur (*redness*) et l'élongation durant la semaine 4;
- D'autres attributs (adhoc, MatchFilter, PCA, etc.) durant le
  premier projet.

L'idée de cette section est de réappliquer ces attributs sur vos
nouvelles données.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b837b5a30e743e46197d299a5491c496", "grade": false, "grade_id": "cell-8658d6e5b24407f1", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Filtres

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8eb49779550fce2e489a45efd16260c4", "grade": false, "grade_id": "cell-ed19444303560eac", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Récapitulons les types de filtres que nous avons en magasin:

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a31f5603e9f34b5e7c0372b9620d4269", "grade": false, "grade_id": "cell-e32ece543b8553b8", "locked": true, "schema_version": 3, "solution": false, "task": false}}

#### La rougeur (redness)

Il s'agit d'un filtre de couleur qui extrait la différence entre la
couche rouge et la couche verte (R-G). Lors de la Semaine 2, nous
avons fait la moyenne de ce filtre sur les fruits pour obtenir un
attribut (valeur unique par image). Ici, affichons simplement la
différence `R-G` :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 56763238cf676a330c25b76485d635f9
  grade: false
  grade_id: cell-c3b13a4e3c38e9e7
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid([redness_filter(img)
            for img in clean_images])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "17eb7024bd19c13089cce0566fa5b150", "grade": false, "grade_id": "cell-d0644350d0879a9d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

À quelles couleurs correspondent les zones claires resp. sombres?
Pourquoi le fond n'apparaît-il pas toujours avec la même clarté?

:::

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b952dd9e20ee6bdbe287e64be9b1ed86", "grade": true, "grade_id": "cell-2ba20f803e981265", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

Les zones claires correspondent aux pixels plus rouges que verts (R > G), et les zones sombres correspondent aux pixels plus verts que rouges. Pour les téléphones allumés, l'écran montre des images avec des couleurs variées, donc on voit des zones claires et sombres selon la couleur affichée. Pour les téléphones éteints, l'écran est noir (R=G= B= 0), donc la différence entre R et G est proche de zéro car elles ont quasiment la même valeur. Le fond apparaît avec une clarté qui change selon sa couleur propre (bois, textile, etc.).

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b45a7b66bebf5249b0188c385c6c7d03", "grade": false, "grade_id": "cell-c285f4ed7c06c6fb", "locked": true, "schema_version": 3, "solution": false, "task": false}}

#### Variante de la rougeur

Il s'agit d'un filtre de couleur qui extrait la rougeur de chaque
pixel calculée avec $R-(G+B)/2$ :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 83d4cf5f89225ae6d5f563e08014b428
  grade: false
  grade_id: cell-878582633eda952d
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid([difference_filter(img)
            for img in clean_images])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4623548f6dc451a2facc4636bd357552", "grade": false, "grade_id": "cell-e60d97b83d7a3fb5", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Pour d'autres idées de mesures sur les couleurs, consulter [cette page
wikipédia](https://en.wikipedia.org/wiki/HSL_and_HSV).

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b4cd104e72053c89ff4903059a057e4e", "grade": false, "grade_id": "cell-743271b9cf06d0e1", "locked": true, "schema_version": 3, "solution": false, "task": false}}

#### Seuillage

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "5c92cd9e375f0960e7c0ba0343cf1314", "grade": false, "grade_id": "cell-caf1f18c7fe5f505", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Souvenez vous que vous pouvez également seuiller les valeurs des
pixels (par couleur ou bien par luminosité). C'est ce que l'on fait
dans les fonctions `foreground_filter` ou `foreground_color_filter`.

:::{admonition} Indication
:class: hint

N'oubliez pas de convertir les images en tableau `numpy` pour
appliquer les opérateurs binaires `<`, `>`, `==`, etc.

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 21ad23af4c65c78983f723e50cbbb9d1
  grade: false
  grade_id: cell-6774ff1e2a9696bb
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid([np.mean(np.array(img), axis = 2) < 100 
            for img in clean_images])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "52a984e5f0ce91d72dfc9a49c086d297", "grade": false, "grade_id": "cell-17090d1991f77f23", "locked": true, "schema_version": 3, "solution": false, "task": false}}

#### Contours

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6130f4caa16deafe9e1da95fa3b7c1a5", "grade": false, "grade_id": "cell-2ee4d4b6672cbfff", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Pour extraire les contours d'une image (préalablement seuillée), on
doit soustraire l'image seuillée avec elle même en la décalant d'un
pixel vers le haut (resp. à droite). Vous l'avez calculé en Semaine 6 et pour le projet2, on fourni cette extraction de
contours avec la fonction `contours` :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 8b7cf2f62226479a4f77c8007cef5369
  grade: false
  grade_id: cell-102b63a88cde7374
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid([contours(np.mean(np.array(img), axis = 2) < 100 ) 
            for img in clean_images])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "13579e20c6b172c1b792ced203613da9", "grade": false, "grade_id": "cell-ff6e22bb84fc2e5f", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Création d'attributs à partir des filtres

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c1354f5844b26180263063c952e06213", "grade": false, "grade_id": "cell-2d2198862d185cd5", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Maintenant que nous avons récapitulé les filtres en notre possession,
nous allons calculer un large ensemble d'attributs sur nos images. Nous vous fournissons un ensemble de fonctions dans utilities qui calcule ces attributs. Une
fois cet ensemble recueilli, nous allons ensuite sélectionner
uniquement les attributs les plus pertinents.

On se propose d'utiliser trois attributs sur les couleurs:

1. `redness` : moyenne de la différence des couches rouges et vertes
   (R-G), en enlevant le fond avec `foreground_color_filter`;
2. `greenness` : La même chose avec les couches (G-B);
3. `blueness` : La même chose avec les couches (B-R).

Ainsi que trois autres attributs sur la forme:

4. `elongation` : différence de variance selon les axes principaux des
   pixels du fruits (cf Semaine2);
5. `perimeter` : nombre de pixels extraits du contour;
6. `surface` : nombre de pixels `True` après avoir extrait la forme.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e4fe2abf2a923c1d7be6cbd93a596a70", "grade": false, "grade_id": "cell-47f4de897fca2092", "locked": true, "schema_version": 3, "solution": false, "task": false}}

::::{admonition} Exercice

Créez la table `df_features` qui contient ces six paramètres,
les attributs que vous avez implémenté dans le projet précédent ainsi que de nouveaux attributs que vous jugerez pertinents pour votre projet.

:::{admonition} Indications
:class: hint

Dans le Projet 1, nous avions défini la table ainsi:

:::

::::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 903dbaa2356b975c09626ebbc9e7e532
  grade: false
  grade_id: cell-bed895de20c6450b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df = pd.DataFrame({
        'redness':    clean_images.apply(redness),
        'elongation': clean_images.apply(elongation),
        'class':      clean_images.index.map(lambda name: 1 if name[0] == 'a' else -1),
})
df
```

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 36e8fc419e3c057c1872f0066de8ed0e
  grade: false
  grade_id: cell-e185754c43dc2d17
  locked: false
  schema_version: 3
  solution: true
  task: false
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "de0849f61e400b17aa436b77eadc2e4a", "grade": false, "grade_id": "cell-dea972a044f9aaa7", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Les amplitudes des valeurs sont très différentes. Il faut donc
normaliser ce tableau de données (rappel de la semaine 3 et du projet 1) afin que les moyennes des colonnes soient égales à 0 et les
déviations standard des colonnes soient égales à 1.

***Indication:*** notez l'utilisation d'une toute petite valeur epsilon
pour éviter une division par 0 au cas où une colonne soit constante :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: fd63701f8e77eec580b7ff122e7b060e
  grade: false
  grade_id: cell-674aab215fb89b4d
  locked: true
  schema_version: 3
  solution: false
  task: false
---
epsilon = sys.float_info.epsilon
df_class = df_features['class']
df_features = (df_features - df_features.mean())/(df_features.std() + epsilon) # normalisation 
df_features.describe() # nouvelles statistiques de notre jeu de donnée
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "022a3326e658b69213f8daff28358451", "grade": false, "grade_id": "cell-53b4be052dea2ca4", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On ajoute nos étiquettes (1 pour la premiere classe, -1 pour l'autre) dans
la dernière colonne :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 69cf92cfac87421bfd07482e634314f3
  grade: false
  grade_id: cell-98e267993504172d
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_features["class"] = df_class
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0f3f21cb3651113e2d643831bae8afcc", "grade": false, "grade_id": "cell-c3c83a907ce1aa73", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Et on remplace les NA par des 0, par défaut:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 4a0509d9fd42ca3373ab797d062f5fa5
  grade: false
  grade_id: cell-cc9c6b17fd135446
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_features[df_features.isna()] = 0
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 049476ba4fc37a04c96f37b5e847d35c
  grade: false
  grade_id: cell-f2756c09cdbe187c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_features.style.background_gradient(cmap='coolwarm')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "cae6d206a6046f304acdd92765942ec4", "grade": false, "grade_id": "cell-a94e7fffe8b34be2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Calcul de performance du 3-NN

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "29c77c71d8661c0ac6dbc64f7d7c2954", "grade": false, "grade_id": "cell-070c3900fd6155c2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous stockerons les performances
d'un classificateur -- ici du plus proche voisin kNN -- dans une table de données `performances`.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: e5bb6ad3110ad37d69f56b646c175f2c
  grade: false
  grade_id: cell-601593426064cfd7
  locked: true
  schema_version: 3
  solution: false
  task: false
---
performances = pd.DataFrame(columns = ['Traitement', 'perf_tr', 'std_tr', 'perf_te', 'std_te'])
performances
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e76e57e5de8e2dff7c5c0f1e51384fdc", "grade": false, "grade_id": "cell-84a3e1e3cf196572", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Importez le modèle KNN avec trois voisins dans l'objet `sklearn_model`.

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 5fe4967ee06b201fca25218dc35d63ba
  grade: false
  grade_id: cell-7d8fd2df1a4699e3
  locked: true
  schema_version: 3
  solution: false
  task: false
---
from sklearn.neighbors import KNeighborsClassifier
# On importe une mesure de précision appelée balanced accuracy score, qui pondère la précision selon la taille du jeu de données
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
```

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 5093325c8e9e289986088235ce62bb50
  grade: true
  grade_id: cell-ed4a2f91f3ee6b60
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
# On importe le  modèle KNN avec 3 voisins
sklearn_model = KNeighborsClassifier(n_neighbors=3)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d4e26eb8afdfca25dc03468692a5cbae", "grade": false, "grade_id": "cell-0c149f31d79c73c7", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Comme dans les séances précédentes, on calcule la performance et les
barres d'erreurs du classifieur par validation croisée (*cross
validate*), en divisant de multiples fois nos données en ensemble
d'entraînement et en ensemble de test.

:::{admonition} Exercice

Finissez de remplir la fonction `df_cross_validate` d'`utilities.py`
afin qu'elle automatise tout le processus. Il y a deux endroits (et 3
lignes) où il y a du code manquant. Consultez son code pour retrouver
les étapes:

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 1493917e01b0751ae0e9534d683aefc8
  grade: false
  grade_id: cell-8830ed861ff8a848
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(df_cross_validate)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e64234390a488b8a71b7976e0a8ed2a3", "grade": false, "grade_id": "cell-f2182f636dd00352", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On calcule ensuite les performances de notre classificateur sur les attributs et on les
ajoute à notre table `performances`.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: ef2c0404f4ab884e45c1b089b3ecacd3
  grade: false
  grade_id: cell-c478d620f43d863f
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# Calcul des performances avec cross validate
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
metric_name
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: e8489340f1d624a1b31fed1340ca8937
  grade: false
  grade_id: cell-0cf01cf4e470c27e
  locked: true
  schema_version: 3
  solution: false
  task: false
---
performances.loc[0] = ["6 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c2d817e3f785737ee442eeb348fee989", "grade": false, "grade_id": "cell-4d940c8d994be5a3", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Sélection des attributs (**nouveau!**)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "65ea456bd9b7fc067fbbad15649d175a", "grade": false, "grade_id": "cell-64b949f4d23c2eb0", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Maintenant que nous avons extrait un ensemble d'attributs, nous
souhaitons analyser lesquels améliorent le plus les performances de
notre classificateur. Pour cela, nous tenterons deux approches :

- **Analyse de variance univariée** : On considère que les attributs
  qui, pris individuellement, corrèlent le plus avec nos étiquettes
  amélioreront le plus la performance une fois groupés.
- **Analyse de variance multi-variée** : On considère qu'il existe un
  sous-ensemble d'attributs permettant d'améliorer davantage les
  performances que les attributs étudiés séparément.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0882c7254b14069b0cf35cb9a2ab7724", "grade": false, "grade_id": "cell-84a65b446afad5f2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Analyse de variance univariée

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "333823e5cfcc749299a829f52075b641", "grade": false, "grade_id": "cell-c03bea43c3f668f2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans cette approche, on commence par calculer les corrélations de
chacun de nos attributs avec les étiquettes :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 51853c2fc00736755150720c4f6b016e
  grade: false
  grade_id: cell-851a22de4ae157ac
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Compute correlation matrix
corr = df_features.corr()
corr.style.format(precision=2).background_gradient(cmap='coolwarm')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "25420e53531b6a9b4441b0241d8f5dd3", "grade": false, "grade_id": "cell-5528d4e1f55152fa", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous allons rajouter de nouveaux attributs sur les
couleurs afin d'identifier s'il y aurait un attribut qui corrélerait
davantage avec les étiquettes.

**NB**: On en profite pour renormaliser en même temps que l'on ajoute
des attributs.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ca3c385819fd13bd8aaa69302bf0e267", "grade": false, "grade_id": "cell-d42fac6b12b6720a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On utilise la fonction `get_colors()` donnée dans utilities

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: a677013657ec89571a4798d27eed3f86
  grade: false
  grade_id: cell-8131a375f984abbc
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(get_colors)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b26552d33812061aa8c7457a37aca236", "grade": false, "grade_id": "cell-efb894fcbb34e174", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Appliquez `get_colors()` à vos images `clean_images`.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: bf27eb0e141693e3c639ec8bcb3f1200
  grade: false
  grade_id: cell-5bd09201a8b4bd74
  locked: false
  schema_version: 3
  solution: true
  task: false
---
df_color=clean_images.apply(get_colors)
df_color
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9a507b882df64ca15aefbd4702f439e7
  grade: false
  grade_id: cell-e8a77493164c858a
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_features
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 6650eaa20893bb11b565ba856b30ecaf
  grade: false
  grade_id: cell-7a8d945a628ee001
  locked: true
  schema_version: 3
  solution: false
  task: false
---
#Voici les nouveaux attributs qu'on ajoute 
# 'R','G','B','M=maxRGB', 'm=minRGB', 'C=M-m', 'R-(G+B)/2',
# 'G-B', 'G-(R+B)/2', 'B-R', 'B-(G+R)/2', 'R-G', '(G-B)/C',
# '(B-R)/C', '(R-G)/C', '(R+G+B)/3', 'C/V' avec V = (R+G+B)/3.

# Première étape: on initialise df_features_large avec les attributs qu'on avait déjà calculé
df_features_large = df_features.drop("class", axis = 1)
# Deuxième étapeon ajoute les nouveaux attributs issus de get_colors()
df_features_large = pd.concat([df_features_large, clean_images.apply(get_colors)], axis=1)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7b6b8939d6c04cbe3e719f31e9400b2e", "grade": false, "grade_id": "cell-982683b31329f838", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Normalisez `df_features_large`. On utilisera un epsilon au cas où certaines valeurs de std() soit nulles.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 9ad3e52bc39c489141b7f264bcab320b
  grade: false
  grade_id: cell-63a6570df5a2cfe2
  locked: false
  schema_version: 3
  solution: true
  task: false
---
epsilon = sys.float_info.epsilon # epsilon

df_features_large = (df_features_large - df_features_large.mean()) / (df_features_large.std() + epsilon)
df_features_large
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3fabd3ee5c0c3b376fe13e2bac9cfcc4", "grade": false, "grade_id": "cell-ab6144ffbb8ba5a6", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Remplacez les valeurs de NA par des 0 puis rajouter les étiquettes (1
pour la première classe, -1 pour l'autre) dans la dernière colonne (sans
normalisation!)

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 2bb7ff623b6fb2dc250b9ec410a04093
  grade: false
  grade_id: cell-5cf962af903dc619
  locked: false
  schema_version: 3
  solution: true
  task: false
---
df_features_large[df_features_large.isna()] = 0
df_features_large["class"] = clean_images.index.map(lambda name: 1 if name[0] == 'a' else -1) # D'après ce qu'on avait fait et donné au dessus
df_features_large
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8e7461339f2be6b37214611c78e6434c", "grade": false, "grade_id": "cell-01264880bfe0fb0c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On vérifie les performances de notre classificateur sur ce large
ensemble d'attributs et on les ajoute à notre table `performances` :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: fe03ed6100d7b5645ffd2bd3ba6e9fe9
  grade: false
  grade_id: cell-02093f583ba285cf
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Validation croisée (LENT)
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_large, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 6fa4c2ccb721558da40351b38e7d8d5a
  grade: false
  grade_id: cell-c603be191196eab2
  locked: true
  schema_version: 3
  solution: false
  task: false
---
performances.loc[1] = ["23 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b13fb51af134dd39ababef32a80c0d58", "grade": false, "grade_id": "cell-aa559e5db97a48c8", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Calculez la matrice de correlation comme on a fait plus haut.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: a8fab7ced6e1e89b1f73fc02b854fb14
  grade: false
  grade_id: cell-33e34bbffac3ecf9
  locked: false
  schema_version: 3
  solution: true
  task: false
---
corr_large = df_features_large.corr()
corr_large.style.format(precision=2).background_gradient(cmap='coolwarm')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d332757810d4a16fe5bb6404f38036b3", "grade": false, "grade_id": "cell-bca813be79b87807", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans l'approche univariée, les attributs qui nous intéresse le plus
sont ceux qui ont une grande corrélation en valeur absolue avec les
étiquettes. Autrement dit, les valeurs très positives (corrélation) ou
très négatives (anti-corrélation) de la dernière colonne sont
intéressants pour nous.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "33d7ee18b632343f7052892cfbf35e58", "grade": false, "grade_id": "cell-d5acd45db38cea70", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On va donc ordonner les attributs qui corrèlent le plus avec nos
étiquettes (en valeur absolue) :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 500a9f8647a7114de2aff5369129cce3
  grade: false
  grade_id: cell-7c3e67dfcca6f41c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Sort by the absolute value of the correlation coefficient
sval = corr_large['class'][:-1].abs().sort_values(ascending=False)
ranked_columns = sval.index.values
print(ranked_columns) 
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "959f919b65cf38439ab7041bfecd3c40", "grade": false, "grade_id": "cell-4a9f0260493fe6a0", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Sélectionnons seulement les trois premiers attributs et visualisons
leur valeurs dans un pair-plot :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 83254613519671656824b5a8c0a336b5
  grade: false
  grade_id: cell-e08b03acd62683c6
  locked: true
  schema_version: 3
  solution: false
  task: false
---
col_selected = ranked_columns[0:3]
df_features_final = pd.DataFrame.copy(df_features_large)
df_features_final = df_features_final[col_selected]

df_features_final['class'] = df_features_large["class"]
g = sns.pairplot(df_features_final, hue="class", markers=["o", "s"], diag_kind="hist")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8a70a137e49d48782ef60560ba90521d", "grade": false, "grade_id": "cell-7002dd8fbb10e127", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Trouver le nombre optimal d'attributs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4de3c160f528909b8e120433b7916ff2", "grade": false, "grade_id": "cell-2a2b1eb294872e13", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On s'intéresse à présent au nombre optimal d'attributs. Pour cela, on utilise la fonction `feature_learning_curve` qui trie les attributs par corrélation puis ajoute progressivement les features dans l’ordre. Elle renvoie les performances en rajoutant les attributs dans l'ordre du
classement fait dans la sous-section précédente (classement en
fonction de la corrélation avec les étiquettes).

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: ae5d20fa5d8d9c793a79b49f962cfb11
  grade: false
  grade_id: cell-1702d101b9aaa0eb
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(feature_learning_curve)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 84edfa1978e33c2524cdc1fba41dfbb7
  grade: false
  grade_id: cell-e7597615c5ad6d84
  locked: true
  schema_version: 3
  solution: false
  task: false
---
feat_lc_df, ranked_columns = feature_learning_curve(df_features_large, sklearn_model, sklearn_metric)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: fdbaf2dce314fb31c46258619c2f471b
  grade: false
  grade_id: cell-b5e1e0f4d421c2d3
  locked: true
  schema_version: 3
  solution: false
  task: false
---
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_tr'], yerr=feat_lc_df['std_tr'], label='Training set')
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_te'], yerr=feat_lc_df['std_te'], label='Test set')
plt.xticks(np.arange(1, 22, 1)) 
plt.xlabel('Number of features')
plt.ylabel(sklearn_metric.__name__)
plt.legend(loc='lower right');
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f2f4f325d0f0d36f930898a3d9a7baa8", "grade": false, "grade_id": "cell-4996b8b770bbb95c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Combien d'attributs pensez-vous utile de conserver?
Justifiez.

:::

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c2afcae45eef66c1bf7667b6ded0a8fd", "grade": true, "grade_id": "cell-24ae5ceb0f20324b", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

D'après la courbe d'apprentissage en fonction du nombre d'attributs, on constate que les performances de test augmentent jusqu'à 4-7 attributs (~0.95 de balanced accuracy) jusqu'à presque atteindre le maximum du pic test qui va sur la courbe de l'entraînement, puis se dégradent clairement au delà de 10 attributs (Surement à cause du surapprentissage : le modèle mémorise les données d'entraînement mais généralise mal). 

Nous conserverons les 7 premiers attributs avec le plus de corrélation.

Soit, nos attributs  : M, C, B,(R+G+B)/3 brightness, filtre_eteint et filtre_allumé.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a754cfb8c7537c534f91155332a9d955", "grade": false, "grade_id": "cell-9006ed5128fae6b4", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Exportez votre table `df_features_final` dans le fichier
`features_data.csv` contenant les attributs utiles. Pour l'exemple ci-dessous,
nous considérerons avoir exporté les trois premiers attributs :

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 2f8d3f1c8dc5427fcff1d2a6d785cef9
  grade: false
  grade_id: cell-a4b1b56535d0b411
  locked: false
  schema_version: 3
  solution: true
  task: false
---
col_selected = ranked_columns[0:3]
df_features_final = df_features_large[list(col_selected) + ['class']].copy()
# Exportation vers CSV
df_features_final.to_csv("features_data.csv")

df_features_final
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9f1f01dfa8975125a80ce64c72561ecd", "grade": false, "grade_id": "cell-e65607c5d0372c04", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Enfin, on ajoute les performance de notre classificateur sur ce
sous-ensemble d'attributs sélectionnées par analyse de variance
univariée et on les ajoute à notre tableau de données `performances` :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 19ce9404b6581952c31e6d7785b84d39
  grade: false
  grade_id: cell-bb10bc28a727b6d0
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Validation croisée
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_final, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 001c8e77bca94eed5a93e4f672189bfc
  grade: true
  grade_id: cell-f9293553bc3eb23b
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
performances.loc[2] = ["3 meilleurs attributs (analyse univariée)", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "17f14be9751aa2eb6b9c1a36902de0aa", "grade": false, "grade_id": "cell-f8492a11c05e01bc", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### ♣ Analyse de variance multi-variée

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "14c4ab8a928f9b5d852804257c517aac", "grade": false, "grade_id": "cell-d21fa0397d45598e", "locked": true, "schema_version": 3, "solution": false, "task": false}}

La seconde approche est de considérer les attributs de manière groupés
et non pas par ordre d'importance en fonction de leur corrélation
individuelle avec les étiquettes. Peut-être que deux attributs, ayant
chacun une faible corrélation avec les étiquettes, permettront une
bonne performance de classification pris ensemble.

Pour analyser cela, on considère toutes les paires d'attributs et on
calcule nos performances avec ces paires. Si le calcul est trop long,
réduisez le nombre d'attributs à considérer.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: cffda426d8a3d2e94ab1ec957988ce3d
  grade: false
  grade_id: cell-182a5422b18315aa
  locked: true
  schema_version: 3
  solution: false
  task: false
---
best_perf = -1
std_perf = -1
best_i = 0
best_j = 0
nattributs = 23
for i in np.arange(nattributs): 
    for j in np.arange(i+1,nattributs): 
        df = df_features_large[[ranked_columns[i], ranked_columns[j], 'class']]
        p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_large, sklearn_model, sklearn_metric)
        if p_te > best_perf: 
            best_perf = p_te
            std_perf = s_te
            tr_best = p_tr
            tr_std = s_tr
            best_i = i
            best_j = j
            
metric_name = sklearn_metric.__name__.upper()
print('BEST PAIR: {}, {}'.format(ranked_columns [best_i], ranked_columns[best_j]))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, best_perf, std_perf))
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, tr_best, tr_std))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2cdd09862ab1ccd5629562e9f81c202a", "grade": false, "grade_id": "cell-b372215747c5fe34", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Ajouter la performance de cette paire d'attributs à la table `performances`.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: f04564c6cc40ff1bc489fb7e122343f1
  grade: true
  grade_id: cell-48d2ee1071945621
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
# Performance paire d'attribut à la table performance
performances.loc[3] = [f"LA Meilleure paire ({ranked_columns[best_i]}, {ranked_columns[best_j]})",
tr_best, tr_std, best_perf, std_perf] 
# on a utilisé un f string pour pouvoir mettre directement le best i et le best j selon leur valeur possible.

performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e9ec834f1ab664203124a98f1415ba8a", "grade": false, "grade_id": "cell-3bc0c874f5b7944c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Quelle est la paire d'attributs qui donne les meilleurs performances?
Est-ce que l'approche multi-variée est nécessaire avec votre jeu de données?

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: e652b379362bfa03cf695be9d9b10598
  grade: true
  grade_id: cell-da9ce91479dfe562
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
# Affichage de la meilleure paire identifiée
print(f"LA Meilleure paire d'attributs : {ranked_columns[best_i]} et {ranked_columns[best_j]}")
print(f"La Performance test : {best_perf:.2f} +/- {std_perf:.2f}")
print("\nL'approche univariée (3 meilleurs attributs) suffit en général pour ce jeu de données car les attributs de luminosité/couleur permettent de très bien distinguer les deux classes.\nL'approche multivariée confirme cela en trouvant la même paire d'attributs dominants.")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6a9647d7892a4e119043c0f227810df0", "grade": false, "grade_id": "cell-f510a13268759d00", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Conclusion

+++ {"deletable": false, "editable": true, "nbgrader": {"cell_type": "markdown", "checksum": "9bbc8f43338562f8c103e2f276e145ca", "grade": true, "grade_id": "cell-c5a8e3d90bxe5375", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Cette feuille a fait un tour d'horizon d'outils à votre disposition
pour le prétraitement de vos images et l'extraction
d'attributs. Prenez ici quelques notes sur ce que vous avez appris,
observé, interprété.

VOTRE RÉPONSE ICI:

Cette feuille nous a permis d'extraire et de sélectionner les meilleurs attributs pour pouvoir classifier les téléphones allumés et éteints. On a pu observé que :

Pour le prétraitement : Le filtre de premier plan  basé sur la rougeur (theta=0.6) permet de bien isoler et extraire le téléphone de son fond (la table), après inversion si le fond est clair et lissage gaussien.

Pour les attributs de couleur: Les attributs de couleur (R, G, B, luminosité V) sont ceux qui arrivent le mieux à dinstinguer et à extraire. Les téléphones allumés ont des écrans brillants (valeurs R, G, B élevées ==> lumiere blanche) tandis que les éteints ont des écrans noirs (RGB: valeurs faibles car noir= absence de couleurs).

Pour les attributs de forme : L'élongation et la surface sont moins discriminants autrement dit, sont les moins efficaces pour reperer et distitnguer les classes car ces classes représentent des téléphones avec la même forme.( Bien que les telephones sont différents pour ceux qui sont éteints.)

Pour la sélection d'attributs : L'analyse univariée montre que 3 attributs suffisent pour atteindre de bonnes performances. L'analyse multivariée va nous permettre de confirmer et de bien valider et verifier le résultat de la première analyse.


Passez à la feuille
sur la [*comparaison de classificateurs*](3_classificateurs.md).

```{code-cell} ipython3

```
