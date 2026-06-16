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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "25b0a055e3f08fdc0f8d3197cc2358cd", "grade": false, "grade_id": "cell-6cfa223d34cccf44", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

# Comparer des classificateurs

Dans la [feuille précédente](3_extraction_d_attributs.md), nous avons
observé que les performances pouvaient varier selon le niveau de
prétraitement des données et les attributs utilisés. Cependant, l'analyse de performance
n'a été conduite qu'avec un seul classificateur (plus proches
voisins).

Il est posssible que chaque classificateur donne des résultats
différents pour un niveau de pré-traitement donné.

Dans cette feuille, on étudie les performances selon le type de
classificateur. Les étapes seront:
1. Déclarer et entraîner les différents classificateurs sur nos
   données;
2. Visualiser les performances de chaque classificateur;
3. Comprendre et identifier le sous-apprentisssage et le
   surapprentissage;
4. Étudier un comité de classificateurs et
   l'incertitude dans les données;
5. **Pour aller plus loin ♣**: Apprentissage profond et transfert
   d'apprentissage.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "03e5d0917d76860a096938f4ea934c38", "grade": false, "grade_id": "cell-88f029d0e00b1e14", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Entraînement des différents classificateurs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ac6554c6da8049e5b0dffa5555a08dfa", "grade": false, "grade_id": "cell-12a48b53cb2b329c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Import des bibliothèques

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: da41b2f7b3362c8c7693c5b328115982
  grade: false
  grade_id: cell-bdc62a79c458052f
  locked: true
  schema_version: 3
  solution: false
  task: false
---
import os, re
from glob import glob as ls
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image
%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings("ignore")
from sys import path

from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4346685a972caed09764f36288d66a7a", "grade": false, "grade_id": "cell-b3011b2b81ff1237", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Import des données

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b177335d943971cee0655fd5fccee414", "grade": false, "grade_id": "cell-f511a65f58aac0d7", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous chargeons les images prétraitées dans la feuille précédente :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 720a30085c1caca7ad1990de11afa760
  grade: false
  grade_id: cell-d8bcf6df946d3631
  locked: true
  schema_version: 3
  solution: false
  task: false
---
dataset_dir = 'clean_data'

images = load_images(dataset_dir, "*.png")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2e0195048a2892e62032f05446e322eb", "grade": false, "grade_id": "cell-543e7be63e4a1b63", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vérifions l'import en affichant les 20 premières images :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 358c7060e8754d3dfc2ff565457d4f64
  grade: false
  grade_id: cell-8cf64e3e7c57bcba
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid(images[:20])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e2c27e4ff8c33ecd484637a207c00612", "grade": false, "grade_id": "cell-33930eb177ebd842", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Pour mettre en œuvre des classificateurs, chargez dans `df_features` les 
attributs extraits dans la fiche précédente.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 6db01de0fd5fd066b5c1b96d0b4092fd
  grade: false
  grade_id: cell-9b7ec7e81dca8f9a
  locked: false
  schema_version: 3
  solution: true
  task: false
---
#On charge les attributs sauvegardés dans la feuille précédente en prenant le fichier csv crée dans la feuille précédente.
df_features = pd.read_csv('features_data.csv', index_col=0)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "bd0fa687a58501d48b18eac08050084a", "grade": false, "grade_id": "cell-4cf16710f27a0f3a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On vérifie que nos données sont normalisées :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 67ad4d45ad5a24d2fdfa2a426eb27bae
  grade: false
  grade_id: cell-fb4a3a55919f5dbc
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_features.describe()
```

```{code-cell} ipython3
# Ici, on voit que Les données sont bien normalisées car les moyennes sont proches de 0 et les écarts-types proches de 1.
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "5342f94306725fddfc30af53878e4240", "grade": false, "grade_id": "cell-a1370cb13b62811d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Déclaration des classificateurs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "30a5e62b3ae3ace075561f17d4636c27", "grade": false, "grade_id": "cell-485190599069b9ed", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous allons à présent importer les classificateurs depuis la librairie
`scikit-learn`. Pour cela, on stockera:
- les noms des classificateurs dans la variable `model_name`;
- les classificateurs eux-mêmes dans la variable `model_list`.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: c461c69ee34d649eb775237f57f22079
  grade: false
  grade_id: cell-7ea31b030159c666
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

model_name = ["Nearest Neighbors", "Parzen Windows",  "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
model_list = [
    KNeighborsClassifier(3),
    RadiusNeighborsClassifier(radius=12.0),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(reg_param=0.1)]
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "59fa4bc0f1c7687b142942bd9727985b", "grade": false, "grade_id": "cell-8bd61b03b2d9ca78", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

La bibliothèque `scikit-learn` nous simplifie la tâche : bien que les
classificateurs aient des fonctionnements très différents, leurs
interfaces sont identiques (rappelez vous la notion d'encapsulation
vue dans le cours «Programmation Modulaire»). Les méthodes sont:
- `.fit` pour entraîner le classificateur sur des données
  d'entraînement;
- `.predict` pour prédire des étiquettes sur les données de test;
- `.predict_proba` pour obtenir des prédictions sur les données de
  test sous forme de probabilités sur les classes;
- `.score` pour calculer la performance du classificateur.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "5db027d943e135a1abfe8bcc3ea13f7f", "grade": false, "grade_id": "cell-7ad47d64e9489ab2", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

## Visualisation des performances des classificateurs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3ff41f00a1c64c1b970e621058a414e7", "grade": false, "grade_id": "cell-0e402814b4b67a62", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Nous allons à présent faire des tests systématiques sur cet ensemble
de données (attributs de l'analyse de variance univiarié). La fonction
`systematic_model_experiment(data_df, model_name, model_list,
sklearn_metric)` permet de réaliser ces tests systématiques.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 77a5bc452230387762fa550ac6eb04bd
  grade: false
  grade_id: cell-f1250a383287f8ca
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
??systematic_model_experiment
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a7d121143da56d819c4fcc37fa2cc8a8", "grade": false, "grade_id": "cell-b75e05e6b7fb13a1", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Que renvoit cette fonction ?

+++ {"deletable": false, "editable": true, "nbgrader": {"cell_type": "markdown", "checksum": "b4a096bf3ddaf0fb0c59a21a62953ebf", "grade": true, "grade_id": "cell-3097cbc02cd7cccb", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

La fonction permet de faire une validation croisée pour les modeles de la liste "model_list" et renvoie un dataframe qui montre les performances moyennes d'entraînement et de test, mais aussi leurs écarts types, pour chaque classificateur. Cela nous permet de comparer plus facilement les classificateurs entre eux et de savoir lequel est le plus efficace.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 27dd3529cbf85954fe1572edd4820fe8
  grade: false
  grade_id: cell-eb3cbeb9d276e186
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
compar_results = systematic_model_experiment(df_features, model_name, model_list, sklearn_metric)
compar_results.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "942233865d9b63b146700d789855276a", "grade": false, "grade_id": "cell-79df935d9f0c8344", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

Quelle méthode obtient les meilleures performances de test?

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 049482cf42bf228c3f073ee945281809
  grade: false
  grade_id: cell-695c5801f831d0ac
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
model_list[compar_results.perf_te.argmax()]
```

+++ {"deletable": false, "editable": true, "nbgrader": {"cell_type": "markdown", "checksum": "c3546817d987a9cf26095d410d344ab8", "grade": true, "grade_id": "cell-b65bf594f7626e4e", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Grâce aux 2 documents ci-dessus, la méthode qui obtient la meilleure perfomance de test est QDA ( QuadraticDiscriminantAnalysis) avec une performance de test de 0.95 et donc la plus haute parmi tous les classificateurs et méthodes.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "11316010316b36d94ad81178756c7228", "grade": false, "grade_id": "cell-961879212e462260", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

On peut également représenter les performances dans un graphique en
barres. Sachant que le nom de la métrique s'obtient avec `sklearn_metric.__name__`, ajouter à la figure le nom des axes pertinents.

:::

```{code-cell} ipython3
---
deletable: false
editable: true
nbgrader:
  cell_type: code
  checksum: 080cd27b9a465d1b052f10649a9a2dc5
  grade: true
  grade_id: cell-1c13f5967f208b71
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
slideshow:
  slide_type: ''
tags: []
---
compar_results[['perf_tr', 'perf_te']].plot.bar()
plt.ylim(0.5, 1)

# VOTRE CODE ICI
plt.xlabel("Modèles")
plt.ylabel(sklearn_metric.__name__)
plt.title("Graphique en barres représentant les performances des modèles / classificateurs")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9eae5f60545a3f343da208837e696935", "grade": false, "grade_id": "cell-ae42ddffdafe2624", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

##  Sous-apprentissage et surapprentissage

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8632d8f026fab6677151290d2b02763b", "grade": false, "grade_id": "cell-7bce4f2d0e3af501", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Lorsque l'on entraîne un classificateur sur des données, deux
comportements sont à éviter:

- Le ***sous-apprentissage*** (*under-fitting*).
- Le ***sur-apprentissage*** (*over-fitting*).

Ces concepts sont vus en [Semaine 8](../Semaine8/CM8.md)


Analysons quels classificateurs ont sur-appris respectivement sous-appris. Pour
cela nous allons identifier les classificateurs dont les *performances de **test*** sont
   inférieures à la *performance de **test** médiane*. `tebad` est un vecteur de booléens.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: f31ee3f5e0efa68ea667b4685586a307
  grade: false
  grade_id: cell-f4aa00764a84d72a
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
tebad = compar_results.perf_te < compar_results.perf_te.median()
tebad
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "bcb6ab0264d30d34e6e43a2ab797b4f8", "grade": false, "grade_id": "cell-7647298fb39a7d9a", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

Calculer maintenant dans `trbad`  les classificateurs dont les performances
   d'entrainement sont inférieurs à la performance d'entrainement médiane. `trbad` est un vecteur de booléens.

:::

```{code-cell} ipython3
---
deletable: false
editable: true
nbgrader:
  cell_type: code
  checksum: 9de2940c08bab41f26dcca421ac07538
  grade: true
  grade_id: cell-502cf2f04f40a8c8
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
slideshow:
  slide_type: ''
tags: []
---
trbad = compar_results.perf_tr < compar_results.perf_tr.median()
trbad
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "56df06cdbf78fa69817bebdd752ba4ca", "grade": false, "grade_id": "cell-674c1cc795d75b15", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

On peut alors identifier les classificateurs qui ont sur-appris (qui correpondent à `True`)
c'est-à-dire ceux dont la performance d'entraînement est **supérieure à la médiane** et 
celle de test est **inférieure à la médiane**. On écrit :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 13891475d4087936c14486f8abcb45f4
  grade: false
  grade_id: cell-925e0e3b353a126b
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
overfitted = tebad & ~trbad
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0955b4f5597dcfcba14098ae47166102", "grade": false, "grade_id": "cell-1f6926f6861f5c4f", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

De la même facon, définir `underfitted` un vecteur de booléens qui indique les classificateurs dont les performances
    d'entrainement et de test sont inférieures à la médiane

:::

```{code-cell} ipython3
---
deletable: false
editable: true
nbgrader:
  cell_type: code
  checksum: 3195a23d5822e431ea8afc2ce154cca0
  grade: true
  grade_id: cell-393d48b50625f575
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
slideshow:
  slide_type: ''
tags: []
---
# VOTRE CODE ICI
# Sous-apprentissage=mauvaises performances en entraînement et test
underfitted = trbad & tebad 
underfitted
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "17dad4f7b27d8a20141d3f0898d9fd57", "grade": false, "grade_id": "cell-221642d5537da55a", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

Implémenter en vous inspirant du code ci-dessus la fonction `analyse_model_experiments` qui:
- prend en entrée les résultats de plusieurs classificateurs sous la forme d'un tableau
- identifie par deux vecteurs de booléens les classificateurs dont les performances de test, respectivement d'entrainement, sont inférieures à la performance de test, resp d'entrainement médiane.
- identifie dans des vecteurs de booléens les classificateurs qui ont sur-appris dans `overfitted` et respectivement sous-appris dans `underfitted`
- ajoute deux colonnes au tableau puis le renvoit
:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 66012a0227948aa05b583e95069b6eb9
  grade: false
  grade_id: cell-bf5de279a6b91aac
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
show_source(analyze_model_experiments)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ea23c1401b5bf67b1a7e6788814e4cfa", "grade": false, "grade_id": "cell-3d2b08fc31dfc5c7", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Puis on applique la fonction à compar_results

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 0dbfd8e74c5b89918f20eadc601305eb
  grade: false
  grade_id: cell-47b4be19fd08b955
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
analyze_model_experiments(compar_results)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "31052686c69de534cf28826ecec44f5b", "grade": false, "grade_id": "cell-c4a46b1532ae7fe4", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

Quelles sont les classificateurs qui ont sous-appris resp. sur-appris?

:::

+++ {"deletable": false, "editable": true, "nbgrader": {"cell_type": "markdown", "checksum": "cda4d2af71e41fea244d7704a5eb8e0a", "grade": true, "grade_id": "cell-9e913181dae0665d", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Le classificateur "Parzen Windows" est le seul qui a sous appris car ses performances d'entraînements et de tests sont mauvais et les classificateurs  "Decision Tree", "Random Forest","AdaBoost" et "Naive Bayes" sont ceux qui ont sur-appris car leurs performances de tests sont relativements mauvaises comparées à celles d'entraînements.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1bcd9f096b652c3c05e903cc38359bb0", "grade": false, "grade_id": "cell-0f2a5f4ad1df888b", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

## Comité de classificateurs et incertitude

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e692af2b4fbdb048b79c421b69b7eb18", "grade": false, "grade_id": "cell-d85f6153b8c45c56", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Un comité de classificateurs est un classificateur dans lequel les
réponses de plusieurs classificateurs sont combinées en une seule
réponse. En d'autres termes, les classificateurs **votent** pour les
prédictions émises.

On pourrait considérer notre liste de classificateur `model_list`
comme un comité de classificateurs, entraînés sur les mêmes données
d'entraînement, et faisant des prédictions sur les mêmes données de
test. Regardez le code suivant; il est en réalité simple: on définit
les méthodes `fit`, `predict`, `predict_proba` et `score` comme
faisant la synthèse des résultats des classificateurs contenus dans
`self.model_list`.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 09c2bcdf35eae0033ce7e88c55895630
  grade: false
  grade_id: cell-4c2fdd613259b27c
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
class ClassifierCommittee():
    def __init__(self, model_list):
        self.model_list = model_list
        
    def fit(self,X,y):
        for model in self.model_list:
            model.fit(X,y)
    def predict(self,X):
        predictions = []
        for model in self.model_list:
            predictions.append(model.predict(X))
        predictions = np.mean(np.array(predictions),axis = 0)
        results = []
        for v in predictions:
            if v < 0:
                results.append(-1)
            else:
                results.append(1)
        return np.array(results)
    
    def predict_proba(self,X):
        predictions = []
        for model in self.model_list:
            predictions.append(model.predict_proba(X))
        return np.swapaxes(np.array(predictions), 0, 1)
    def score(self,X,y):
        scores = []
        for model in self.model_list:
            scores.append(model.score(X,y))
        return np.swapaxes(np.array(scores), 0, 1)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7d1911825fe6724859fbcfe1d62dc02b", "grade": false, "grade_id": "cell-452758ca86801b63", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

On définit un comité à l'aide de ce code:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 07de6e9dc6b8360cc5bbb427252173c9
  grade: false
  grade_id: cell-c41772320064e711
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
commitee = ClassifierCommittee(model_list)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0cebecc4a725f98bc12066643637db73", "grade": false, "grade_id": "cell-0f0d841fcc1b3058", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice 

Quelles seraient les performances d'un tel classificateur?  Calculer les performances de `commitee`

::::

```{code-cell} ipython3
---
deletable: false
editable: true
nbgrader:
  cell_type: code
  checksum: 4251d1c67486b09e0d357fbfa7083799
  grade: true
  grade_id: cell-e2a2cd51a5aec85e
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
slideshow:
  slide_type: ''
tags: []
---
# ON REPREND LA Validation croisée (LENT) de la feuille 2 mais en adaptant avec commitee !!
# On calcule des performances du comité de classificateurs 

p_tr_com, s_tr_com, p_te_com, s_te_com = df_cross_validate(df_features, commitee, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()

print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr_com, s_tr_com))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te_com, s_te_com))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "762e8ae2505c019d1ad004a5c9a67c6e", "grade": false, "grade_id": "cell-7c9cfd87890b021f", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

La performance de l'ensemble de classificateurs n'est pas forcément
meilleure que les classificateurs pris individuellement. Cependant,
l'accord ou le désaccord des classificateurs sur les prédictions peut
nous donner des informations sur l'incertitude des données.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6fae4dc43c94a47c1d04001ebd123eb3", "grade": false, "grade_id": "cell-645f39bd1be4bf65", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

En effet, chaque classificateur peut émettre des probabilités sur les
classes avec la méthode `predict_proba`.  Pour quantifier
l'incertitude d'une image, on peut distinguer deux cas de figure :
- **l'incertitude aléatorique** : les classificateurs sont plus ou moins certains de leur prédiction. 
- **l'incertitude épistémique** : les classificateurs sont d'accord ou pas entre eux. 

Ces incertitudes vont être d'autant plus fortes pour les images proches des frontières de décision (les bananes rouges par exemple).
Intéressons nous aux images avec une faible incertitude épistémique
(les classificateurs sont d'accord) et une grande incertitude
aléatorique (les classificateurs sont incertains de la
prédiction). Cela correspond à des images se situant aux abords de la
frontière de décision entre nos deux catégories.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1700a72f498d5596b4d490a6b6de8a72", "grade": false, "grade_id": "cell-0fea84bc70094418", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

### Incertitude aléatorique

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6f2e581f20f17ffd59c4c5127d96a503", "grade": false, "grade_id": "cell-3f21c655edcfd8ab", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Nous allons utiliser l'[entropie de
Shannon](https://fr.wikipedia.org/wiki/Entropie_de_Shannon), qui est
une mesure en théorie de l'information permettant d'estimer la
quantité d'information contenu dans une source d'information (ici
notre probabilité sur les classes): 

$$H(X) = - \sum_{i=1}^{n}P(x_i)log_2(x_i)$$

où $x_i$ est la probabilité de classification sur la classe $i$.

On récupère alors les prédictions de nos images pour les 11
classificateurs :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 5362c786e5ef2c69491d10d2ee81fb51
  grade: false
  grade_id: cell-3c34e1651bb890d5
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
X = df_features.iloc[:, :-1].to_numpy()
Y = df_features.iloc[:, -1].to_numpy()
commitee.fit(X, Y)
prediction_probabilities = commitee.predict_proba(X)
prediction_probabilities.shape
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2e604742a060b96fee211d4434e5c84d", "grade": false, "grade_id": "cell-0e697b40b4cd7d5c", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

La dimension de notre matrice de prédiction est donc: `(nombre
d'images, nombre de classificateur, nombre de classess)`. Appliquons
l'entropie pour chaque prédiction de chaque classificateur :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 442a78916f4968d2123c742a784a6cf1
  grade: false
  grade_id: cell-1ba73503201dffe9
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
from scipy.stats import entropy
entropies_per_classifier = entropy(np.swapaxes(prediction_probabilities, 0, 2))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "af1d0c57e4e16635efd36827df98e422", "grade": false, "grade_id": "cell-f5c2145fe08fa7d8", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

On moyenne les entropies d'une image entre les classificateurs :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: bb57f6978cfb9233c5ac5eda457fa4e9
  grade: false
  grade_id: cell-73427b1fa71fa9e1
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
entropies = np.mean(entropies_per_classifier, axis = 0)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e362f815120c54d2dcb78e09bd948def", "grade": false, "grade_id": "cell-9679ab6f47fd7bf1", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Puis on ajoute ces valeurs dans une table que l'on trie par ordre
décroissant d'entropie. Les images avec l'entropie la plus grande sont
les plus incertaines et donc les plus informatives pour le modèle :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 0b85d70f51c3d7ff5be3373bf6726704
  grade: false
  grade_id: cell-d220e1fb8963eaa7
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
df_uncertainty = pd.DataFrame({"images" : images,
                           "aleatoric" : entropies})
df_uncertainty = df_uncertainty.sort_values(by=['aleatoric'],ascending=False)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 1d628feba8f099ab3e978887541fcc7e
  grade: false
  grade_id: cell-b55334aaccacdc1e
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
df_uncertainty.style.background_gradient(cmap='RdYlGn_r')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "820858ad5bd4bdb646b3f815eea63e8a", "grade": false, "grade_id": "cell-dd18631ce35028c4", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

Affichons les 10 images **les plus incertaines** pour le comité de
classificateurs, selon cette mesure aléatorique de l'incertitude. Ces
images nous donne une idée de l'ambiguité intrinsèque de notre base de
données.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 8e59de3eaa90738fe47a4eef68bedb38
  grade: false
  grade_id: cell-8876c75845013319
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
uncertain_aleatoric_images = df_uncertainty['images'].tolist()
image_grid(uncertain_aleatoric_images[:10])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2f247af9d02e840566bff80d25de6150", "grade": false, "grade_id": "cell-e978b5155f366bac", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition}

Afficher maintenant les 10 images les **plus certaines** pour
notre comité de classificateurs 

:::

```{code-cell} ipython3
---
deletable: false
editable: true
nbgrader:
  cell_type: code
  checksum: 9cb74c0117b4605ac165ee4fbd77fce8
  grade: true
  grade_id: cell-b763c71e41096556
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
slideshow:
  slide_type: ''
tags: []
---
# Pour faire cela, on va reprendre le data frame au dessus mais en le triant de manière croissante cette fois-ci
# Les 10 images les plus certaines = faible entropie = tri croissant /ascendant = True
df_certainty = df_uncertainty.sort_values(by=['aleatoric'],ascending=True) # Ici true à la place
certain_aleatoric_images = df_certainty['images'].tolist()
image_grid(certain_aleatoric_images[:10])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ac879542035e652bfb437fb023b1a2cb", "grade": false, "grade_id": "cell-a4f0ee22f69b361f", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

Ces résultats vous semblent-ils surprenants? Expliquer.

:::

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1e0e1fab3fe3e4982abd1c7806e45788", "grade": true, "grade_id": "cell-a47dcbcbba79299e", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

Les résultats ne sont pas surprenants.
En effet,les images les plus certaines correspondent aux téléphones dont l'état (allumé ou éteint) est le plus visible et le plus facile à interpreter. 

Pour les téléphones éteints, un écran totalement noir est très facile à classifier. Pour les téléphones allumés avec un fond/ image lumineux et coloré, la classification est aussi simple. 

Les images les plus incertaines sont celles où l'écran présente des tons sombres (fonds et images sombres sur un téléphone allumé) ou des couleurs proches du noir car ils n'arrivent plus à distinguer et à differencier les images avec un teléphone eteint et un télephone qui a un fond lumineux mais sombre.

NB: 

Incertitude aléatorique = liée au bruit ou au caracctère ambigu des donnéess + Impossible à réduire celle ci. Donc l'incertitude est élevée pour des images floues ou ambigues 

Incertitude epistémique: liée au manque de connaissance du modèle + Possible de réduire incerittude avec + de données. Donc, l'incertitude est élévée pour des images nouvelles, atypiques ou jamais vue.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "dcb929b82cad5e239c92ed22403fa8c3", "grade": false, "grade_id": "cell-6b9360942d3fe79e", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

### Incertitude épistémique

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7d5c0d89c7b2c4203c0cee559fdcf2af", "grade": false, "grade_id": "cell-c4382c4bc5b4e599", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Pour l'incertitude épistémique, on va simplement moyenner les écarts types entre  entre les
classificateurs. Comme on n'a que deux classes l'écart type entre les classificateurs pour la première classe est la même que pour la seconde:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: f1a341327c5ee06956f7fe4cbd8704d4
  grade: false
  grade_id: cell-3728f14b1c193eea
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
# std entre les classificateurs
epistemic_uncertainty = np.std(prediction_probabilities[:,:,0], axis = 1)
print(epistemic_uncertainty.shape)

print(epistemic_uncertainty)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d9cb5a4fc442d71b779c26caec54dc45", "grade": false, "grade_id": "cell-04b95b8d4e6c61dd", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

:::{admonition} Exercice

Ajouter cette mesure au tableau `df_uncertainty` .

:::

```{code-cell} ipython3
---
deletable: false
editable: true
nbgrader:
  cell_type: code
  checksum: f730458e4944b4297b5d0ce15609cb58
  grade: true
  grade_id: cell-0b5180d6df21bdeb
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
slideshow:
  slide_type: ''
tags: []
---
df_uncertainty["epistemic"] = epistemic_uncertainty
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 834337bd006f098a1c0df5a0ff2ba7f4
  grade: false
  grade_id: cell-dac2e40e9ec3ff2d
  locked: true
  schema_version: 3
  solution: false
  task: false
slideshow:
  slide_type: ''
tags: []
---
df_uncertainty = df_uncertainty.sort_values(by=['epistemic'],ascending=False)
df_uncertainty.style.background_gradient(cmap='RdYlGn_r')
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 6681d1d9ab7a3e4e0ce1e7c9aac79f6d
  grade: false
  grade_id: cell-265a2aec9d6e94da
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_uncertainty[["aleatoric",'epistemic']].corr()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "396efed5449206d0823fcbaad92cbd3d", "grade": false, "grade_id": "cell-26aeb59af687a719", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Les valeurs d'incertitude aléatorique (entropie) et épistémiques (std)
ne semblent pas très corrélées. Afficher les 10 images les plus
incertaines selon la mesure d'incertitude épistémique :

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: af5bb506d03cef74636cfff21d2424aa
  grade: true
  grade_id: cell-f4b879de9d85049a
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
# On veut les 10 images les plus incertaines selon l'incertitude épistémique
# On reprend la même méthode que ci-dessus
uncertain_epistemic = df_uncertainty.sort_values(by=['epistemic'], ascending=False)
image_grid(uncertain_epistemic['images'].tolist()[:10])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a87c41e2a2a1bdb1795f0d36f41251fd", "grade": false, "grade_id": "cell-5adf2198497793b2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Afficher les 10 images les plus
certaines selon la mesure d'incertitude épistémique :

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 93e5b72dc7c830d233541a055c57c11a
  grade: true
  grade_id: cell-5cf73e64fa0f9ed3
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
# 10 images les plus incertaines selon l'incertitude épistémique
certain_epistemic = df_uncertainty.sort_values(by=['epistemic'], ascending=True)
image_grid(certain_epistemic['images'].tolist()[:10])
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6fbe77d94306114f7f32e8a6d5e8aac7", "grade": true, "grade_id": "cell-d32fa098a68274b0", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

:::{admonition} Exercice

Ces résultats vous semblent-ils surprenants? Expliquer.

VOTRE RÉPONSE ICI

NB: 

Incertitude aléatorique = liée au bruit ou au caracctère ambigu des donnéess + Impossible à réduire celle ci. Donc l'incertitude est élevée pour des images floues ou ambigues 

Incertitude epistémique: liée au manque de connaissance du modèle + Possible de réduire incerittude avec + de données. Donc, l'incertitude est élévée pour des images nouvelles, atypiques ou jamais vue.

Les résultats semblent être cohérents et attendus. L'incertitude épistémique (donc le désaccord entre classificateurs) est plus grande pour les images ambiguës, différents, uniques, i.e les téléphones allumés avec un fond d'écran sombre ou les téléphones éteints dans un environnement lumineux. En effet, ce sont les images qui ont un écart type le plus grand i.e qui sont les plus dispersées de la moyenne des images. Alors, ce sont les images qui difèrent et ne ressemblent pas aux images moyennes, qu'on retrouve de manière redondante. 

Les images les plus certaines pour l'incertitude épistémiques sont celles où tous les classificateurs s'accordent, alors typiquement les téléphones clairement éteints (écran totalement noir) ou clairement allumés (écran lumineux et coloré). 

Il y a aussi une très faible corrélation entre incertitude aléatorique et épistémique d'après le tableau "df_uncertaintly". C'est très interessant car on voit donc qu'un classificateur peut être incertain de sa réponse tout en étant cohérent avec les autres (incertitude aléatorique élevée, épistémique faible) et inversement.



:::

+++

## Conclusion

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "175abe395c45f2d230bcc2e7396363ac", "grade": true, "grade_id": "cell-48ecfb6975c2fe0b", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

Cette feuille a fait un tour d'horizon d'outils de classification à
votre disposition. Prenez ici quelques notes sur ce que vous avez
appris, observé, interprété.

Cette feuille nous a aidé à comparer differents classificateurs pour notre classification téléphones allumés/éteints :

La majorité des classificateurs ont des bonnes performances ( balanced accuracy élevé) grâce aux attributs de luminosité qui permettent de bien distinguer bien les deux classes.

Les classificateurs simples (k-NN, Naive Bayes, Linear SVM) sont plutot efficace avec les modèles plus complexes.
Certains classificateurs font du surapprentissage (arbre de décision profond) : leurs performances en entraînement sont excellentes mais sont mauvais en test.
Le comité de classificateurs permet de bien estimer l'incertitude des prédictions.

L'analyse de nos incertitudes montre que les images les plus difficiles à classifier sont les téléphones allumés avec un contenu sombre, qui ressemblent à des téléphones éteints ou des télephones avec un contenu unique, différent de tous les autres.

+++

Faites maintenant votre propre [analyse](0_analyse.md) et préparez votre [diaporama](0_diaporama.md)!
