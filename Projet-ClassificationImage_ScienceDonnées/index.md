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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a8cd4ae501720003ed96fa2ea305aa2f", "grade": false, "grade_id": "cell-ebb8f94141812ddb", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

# Semaine 7, 8 et 9: mini projet 2

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f3f2d46fae590c69bea03665443c0b13", "grade": false, "grade_id": "cell-ebb8f94141812ddc", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Objectifs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "48f92d8a1abf93e2dee632c9a9ecbe27", "grade": false, "grade_id": "cell-d94d0b397fc433ff", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vous êtes désormais familiers avec le schéma VI-ME-RÉ-BAR --
[VI]sualisation, [ME]sure, [RÉ]férence (baseline), [BAR]res d'erreur,
et vous avez appris à classifier des jeux de données simples:

- Collecter et traiter des images (recentrage automatique, seuillage
  etc.).
- Extraire des attributs sur les couleurs (ad-hoc), la forme (ad-hoc
  ou matched filters) ou les pixels (PCA).
- Entraîner un classificateur (oneR, k-NN, arbre de décision, perceptron
  ou autres) sur vos données d'entraînement.
- Calculer un score de performance du classificateur sur vos données
  de test.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6567aa4bbaabf410c4290f53b62d119b", "grade": false, "grade_id": "cell-3c07cc11be90a0d2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

À l'occasion du second mini-projet, vous allez appliquer ces
connaissances et votre savoir faire à vos propres données, en
développant de nouveaux aspects de l'analyse (biais sur les
méta-données) ou en allant plus loin des les concepts déjà introduits
(PCA, nouveaux classificateurs).

:::{important}

Le travail est à effectuer en binôme, sauf demande motivée auprès de votre chargée ou chargé de TP.

:::

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "89dcb65d81a233d8204719764ef62ca5", "grade": false, "grade_id": "cell-c8a2908965b450a0", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

## Planning

### Semaine 7

- Préparation de vos données (recadrage, extraction de l'objet), première analyse avec l'ACP et
  extraction d'attributs

### Semaine 8

- Retour sur les classificateurs.

### Soutenances en Semaine 9

- La soutenance durera 6min suivies de 2 questions pour chaque membre binôme.
- Le contenu attendu de votre présentation est disponible dans la
  feuille [`0_diaporama.md`](0_diaporama.md). Vous devrez présenter:
    * Votre jeu de données, la motivation de ce choix, 
    * La préparation des données et les attributs
      choisis ou l'ACP (montrez le code, les différentes étapes en image si cela est pertinent),
    * Les résultats (le choix du classificateur, le taux d'erreur etc), 
    * Les difficultés rencontrées, les biais possibles
      de vos images,
    * Une conclusion et des perspectives (à quoi pourrait servir votre
      projet dans notre société ? Comment auriez-vous pu l'améliorer)
- Posez vos questions relatives à la soutenance sur le forum: nous ne
  répondrons pas par mail privé.

## Autres consignes pour le projet

- Votre projet correspond à la feuille [`0_analyse.md`](0_analyse.md). Nous n'irons regarder que cette feuille. 
- Vous mettrez dans le fichier [utilities.py](utilities.py) les
  utilitaires des TP précédents (`load_images`, ...)  que vous
  souhaiterez réutiliser, ainsi que vos nouvelles fonctions.
- Lorsque pertinent pour les explications, vous pourrez afficher le
  code de vos fonctions avec `show_source`.
- Vérifiez à chaque fois le résultat des outils de
  vérifications (`flake8`).
- Lorsque vous aurez besoin de brouillon -- par exemple pour mettre au
  point du code -- créez des petites feuilles Jupyter séparées pour ne
  pas polluer votre document.


## Au travail!

1.  La feuile d'[*analyse*](0_analyse.md) vous servira à faire votre
    analyse de données en intégrant ce que vous avez fait dans les
    feuilles suivantes (ACP ou extraction d'attributs; choix du ou
    des classificateurs etc). Il y a le squelette d'une analyse de données.
2.  Ouvrez la feuille [*jeu_de_donnees*](1_jeu_de_donnees.md) et suivez
    les instructions pour charger votre jeu de données (feuille 
    identique à celle de la Semaine 5 + dépot des images via git)
3.  Ouvrez la feuille [*première analyse
    ACP*](2_premiere_analyse_ACP.md) pour faire une première analyse de
    vos images. Le squelette de cette feuille est adaptée pour le jeu
    de données des pommes et des bananes. Ce code est à modifier pour
    l'adapter à votre projet.
4.  Ouvrez la feuille [*extraction
    d'attributs*](3_extraction_d_attributs.md) et suivez les
    instructions pour prétraiter vos images et en
    extraire des attributs. Le squelette de cette feuille est adaptée pour le jeu
    de données des pommes et des bananes. Ce code est à modifier pour
    l'adapter à votre projet.
5.  Ouvrez la feuille [*classificateur*](4_classificateurs.md) et suivez
    les instructions pour faire de nouveaux types de classification.
    Le squelette de cette feuille est adaptée pour le jeu
    de données des pommes et des bananes. Ce code est à modifier pour
    l'adapter à votre projet.
6.  Préparer votre présentation. Un squelette de [*diaporama*](0_diaporama.md)
    Jupyter vous est fourni; vous pouvez aussi produire un diaporama en pdf.
    Votre présentation devra présenter de façon synthétique votre jeu de données,
    vos expériences, vos observations, vos interprétations.

+++

:::{tip}

Les feuilles Jupyter fournies sont très guidées. Si vous souhaitez
effectuer d'autres traitements et analyses, ou simplement pour faire
des essais, n'hésitez pas à utiliser de nouvelles feuilles Jupyter
libres. De même, vous pouvez ajouter d'autres fichiers, comme des
images pour illustrer votre diaporama.  **Assurez-vous de signaler ces
fichiers supplémentaires à `git` pour qu'ils soient déposés sur Gitlab
avec le reste de votre projet.** 

Pour ajouter un fichier
nommé `truc.pdf` au dossier `Semaine7`:

1) Ouvrir le tableau de bord
2) Récupérer/Déposer la Semaine 7 
3) Ajouter une cellule Jupyter de code et y taper la commande suivante :

    ! cd ~/IntroScienceDonnees/Semaine7; git add truc.pdf

4) Déposer de nouveau la Semaine 7

:::

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

### Travail en binome et dépôts

- [ ] Vérifiez votre inscription en binôme dans le [document
    partagé](https://codimd.math.cnrs.fr/3f98v4-YT4CN2ktM6_g5lw?both).
- [ ] Révisez les [bonnes pratiques](../Semaine3/1_bonnes_pratiques.md)
      vues en semaine 3.

:::{seealso}

Pour en savoir plus et pour des suggestions sur comment travailler en
binôme, veuillez vous référer à la
[page web](https://nicolas.thiery.name/Enseignement/IntroScienceDonnees/ComputerLab/travailBinome.html).
TODO
:::

+++

Bon Projet!
