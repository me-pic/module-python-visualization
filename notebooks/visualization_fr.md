---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python cours visu
    language: python
    name: visu-env
---

<!-- #region editable=true slideshow={"slide_type": ""} -->
# Visualisation et interprétation de modèles


## Objectifs d'apprentissage 
&#x1F440; Comprendre l'utilité des visualisations
<br> &#x1F4C8; Comprendre quel type de graphique utilisé selon le type de données
<br> &#x1F3A8; Choisir adéquatement sa palette de couleurs
<br> &#x1F503; Apprendre comment modifier différents éléments de nos figures
<br> &#x1f9e0; Utiliser `nilearn` pour la visualisation de données de neuroimagerie

## Organisation de la séance

La séance comportera des sections théoriques et des sections pratiques:

**Théorique**

Nous allons discuté des principes théoriques de base en visualisation. Les principes suivants seront couverts:
- Types de graphiques (données tabulaires): univarié vs bivarié; catégoriel vs continue
- Palette de couleurs: perceptuellement uniformes vs non-uniformes; discrètes vs continues
- Types de graphiques (données de neuroimagerie): cartes statistiques, connectome, etc.

Cette partie inclut plusieurs éléments interactifs pour amener les étudiant.e.s à former leur propre compréhension de la matière. Le code est déjà fourni, mais nous ne nous y attarderons pas. 

**Pratique**

Nous allons mettre en pratique, au fur et à mesure, les principes théoriques. Nous allons utiliser les librairies suivantes:
- matplotlib
- seaborn
- ptitprince
- plotly
- nilearn

Les étudiant.e.s devront modifier le code donné pour comprendre le rôle de différents paramètres de visualisation pour répondre à certaines questions à partir de ce que nous allons voir en classe ou à partir de références fournies (p. ex. documentation matplotlib).
<!-- #endregion -->

<div class="alert alert-block alert-warning">
Les <b>encadrés jaunes</b> contiennent des questions/exercices auxquelles les étudiant.e.s doivent répondre.
</div>


<div class="alert alert-block alert-info">
Les <b>encadrés bleus</b> contiennent des informations complémentaires sur les jeux de données et les fonctions utilisés.
</div>

```python editable=true slideshow={"slide_type": ""}
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import ptitprince as pt
import ipywidgets as widgets
import matplotlib.pyplot as plt
import plotly.express as px

from cmcrameri import cm
from nilearn import datasets, plotting, image
from nilearn.input_data import NiftiLabelsMasker
from matplotlib import colormaps, colors
from colorspacious import cspace_converter, cspace_convert
```

## Télécharger les données 

### Jeu de données Brain development fMRI

Nous allons utiliser les données provenant du jeu de données *brain development fMRI* qui inclu les données phénotypiques, ainsi que les données IRMf collectées auprès d'enfants et d'adultes lors d'une tâche de visionnement de film ([Richardson et al., 2018](https://doi.org/10.1038/s41467-018-03399-2)).

**Note:** si vous avez déjà téléchargé vos données et qu'elles ne se trouvent pas dans le dossier <i>data/</i>, modifier le chemin dans la cellule ci-dessous.

```python
development_dataset = datasets.fetch_development_fmri(data_dir='data/')
```

### Datasaurus
Pour la première partie de ce tutoriel, nous allons très brièvement utiliser le jeu de données Datasaurus. Si vous voulez être en mesure d'exécuter les cellules utilisant ce jeu de données, vous devrez le télécharger à partir de [kaggle](https://www.kaggle.com/datasets/tombutton/datasaurusdozen).

**Note:** modifier le chemin dans la cellule si dessous au besoin.

```python
# Credit: Alberto Cairo (original datasaurus), and Justin Matejka and George Fitzmaurice (datasaurus dozen)
data = pd.read_csv("data/datasaurus.csv") 
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Une image vaut mille mots 
- Les visualisations nous aide à mieux comprendre la complexité de nos données
- Elles nous permettent de supporter nos résultats
- Et de partager un message
<!-- #endregion -->

```python
# rcParams permet de modifier les paramètres globaux de nos figures
# Ici, on s'assure simplement que les valeurs entre (-8000000, 8000000) ne seront pas montrées en notation scientifique
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.limits'] = (-8000000, 8000000) 
```

```python
def plot_category(category):
    plt.scatter(data[data['dataset'] == category]['x'], data[data['dataset'] == category]['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    max_x = data[data['dataset'] == category]['x'].max()
    max_y = data[data['dataset'] == category]['y'].max()
    plt.text(max_x-0.22*max_x, max_y+0.10*max_y, f"Mean: ({data[data['dataset'] == category]['x'].mean().round(2)}, {data[data['dataset'] == category]['y'].mean().round(2)})")
    plt.text(max_x-0.22*max_x, max_y+0.06*max_y, f"Std: ({data[data['dataset'] == category]['x'].std().round(2)}, {data[data['dataset'] == category]['y'].std().round(2)})")

    plt.show()

widgets.interact(
    plot_category,
    category=widgets.Dropdown(
        options=sorted(data['dataset'].unique()),
        description='Dataset:'
    )
);
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### "Ne faites jamais confiance aux statistiques descriptives, visualisez toujours vos données !"
<p style="margin-top: -1em;"><i>Albert Cairo, créateur du jeu de données Datasaurus</i></p>

Python nous offre une panoplie de librairie pour visualiser nos données:
- Haut niveau vs bas niveau
- Images statiques vs graphiques interactifs
- Plusieurs librairies générales (p.ex., matplotlib, seaborn, bokeh et plotly)
- Et spécifiques à un domaine (p. ex., nilearn)

Mais avec un grand pouvoir viennent de grandes responsabilités...
<!-- #endregion -->

```python
data = pd.DataFrame(
    {
        'Categories': ['A', 'B'],
        'Values': [6000000, 7066000]
    },
    
)
```

```python
plt.bar(data['Categories'], data['Values'])
plt.ylim([5500000, 8000000])
plt.yticks([])
plt.title("Que pouvez-vous dire sur 'A' et 'B' si vous vous fiez uniquement à ce que vous voyez dans la figure ?")
plt.show()
```

```python
plt.bar(data['Categories'], data['Values'])
plt.ylim([5500000, 8000000])
    
plt.title("Que remarquez-vous ?")
plt.show()
```

```python
plt.bar(data['Categories'], data['Values'])

# Ajoute les valeurs au-dessus des bars
for i, v in enumerate(data['Values']):
    plt.text(i, v + 1, str(v), ha='center', va='bottom')
    
plt.title("Mieux ?")
plt.show()
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## À chaque type de données son type de graphique !

Il existe plusieurs types de graphiques. Le type de graphique choisi dépend des variables que l'on veut visualiser:
- Visualisation univariée: **variable continue** vs **variable catégorielle**
- Visualisation bivariée: **catégorielle x catégorielle** vs **catégorielle x continue** vs **continue x continue**
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Visualisations univariées

Pour une **variable continue**, nous pouvons visualiser sa distribution:
- Histogramme
- Estimation de la densité par noyau (*kde plot*)
- Nuage de points univarié (*strip plot*)
  
Pour une **variable catégorielle**, nous pouvons visualiser la quantité d'observations pour chaque catégorie:
- Diagramme à barres (*bar plot*)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
<div class="alert alert-block alert-danger">
Si vos données ne se trouvent pas dans le dossier <i>data/</i>, modifier le chemin dans la cellule ci-dessous.
</div>
<!-- #endregion -->

```python
# Allons chercher les données
participants = pd.read_csv('data/development_fmri/development_fmri/participants.tsv', sep='\t')

# Regardons ce que nous avons dans nos données
...
```

```python
# Regardons le type de nos données
participants.dtypes
```

<div class="alert alert-block alert-info">
Le jeu de données contiennent des variables continues (p.ex. `Age`, `ToM Booklet-Matched`, `FB_Composite`) et des variables catégorielles (p.ex. `AgeGroup`, `Child_Adult`, `Gender`). `ToM Booklet-Matched` représente un score à une tâche visant à évaluer la théorie de l'esprit, c'est-à-dire la capacité à attribuer des états mentaux (croyances, désirs, émotions, intentions) à soi-même ou aux autres.
</div>

```python
# Regardons d'abord les statistiques descriptives associées à la variable continue `Age`
print(participants['Age'].describe())
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
#### Histogramme

Un histogramme nous permet de visualiser la distribution d'une variable donnée de manière discrète en groupant ses valeurs dans des intervalles consécutifs (bins). On obtient donc la fréquence (c.-à-d. le nombre d'observations) dans chacun de ces intervalles.

Il est possible de générer un histogramme dans matplotlib grâce à [la fonction `hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
<!-- #endregion -->

```python
# Regardons maintenant sa distribution
...
# Ajoutons un titre (title)
...
# Ajoutons un titre à l'axe des x (xlabel) et des y (ylabel)
...
...
```

<div class="alert alert-block alert-info">
<b>La science derrière le nombre de bins à visualiser - partie 1</b>
<br>La fonction <i>hist</i> dans <i>matplolib</i> groupe les données en 10 bins par défaut. Par contre, un nombre de bins inadéquat (soit trop petit, soit trop grand) ne montre pas la distribution de manière représentative.
</div>


<div class="alert alert-block alert-warning">
<b>Le nombre de bins en pratique !</b>
<br>Modifier la valeur du paramètre `bins` dans la cellule ci-dessous pour déterminer le nombre de bins qui représenterait le mieux notre variable `Age`
</div>

```python
# Modifier la valeur du paramètre `bins` (ex. bins=10)
...
```

<div class="alert alert-block alert-info">
<b>La science derrière le nombre de bins à visualiser - partie 2</b>
<br> Il existe différentes règles permettant de calculer le nombre de bins optimal, ainsi que l'intervalle à utiliser pour chacune de ces bins. Il est possible d'utiliser ces règles à même la fonction <i>hist</i> de matplotlib ! Il ne suffit que de préciser une des règles valides au paramètre `bins`.
</div>

```python
help(plt.hist)
```

```python
# Regardons maintenant sa distribution (ex. bins='fd')
...
```

#### Estimation de la densité par noyau (*kde plot*)

L'**estimation de la densité par noyau** (ou *kernel density estimation*) nous permet de visualiser la distribution de notre variable de manière continue (plutôt que discrète) en estimant une fonction de densité. Par contre, similairement à la taille des bins pour l'histogramme, l'estimation de la densité par noyau est sensible à la largeur du noyau !

```python
# Pour visualiser l'estimation de la densité par noyau, nous allons utiliser la fonction
# `kdeplot` dans la librairie `seaborn`

...
```

<div class="alert alert-block alert-info">
<b>L'axe des y sur un kde plot - partie 1</b>
<br> Vous avez peut-être remarqué que les valeurs sur l'axe des y vont maintenant de 0 à 0.08 (vs 0 à 50 pour notre histogramme). Dans un kde plot, l'axe des y représente la densité, c'est-à-dire la probabilité par unité de la variable sur l'axe des x. En d'autres mots, à quel point les données sont denses pour une valeur de x donnée. Donc, les pics dans ce type de figure représentent une plus grande densité de points pour un étendu de valeurs donné (c.-à-d. une plus grande probabilité qu'une valeur soit observée), alors que les creux représentent une plus faible densité de points.
</div>


<div class="alert alert-block alert-info">
<b>À noter !</b>
<br>Puisque ce type de graphique fourni une estimation continue, nous pourrions penser que certaines données existent alors qu'en réalité de n'est pas le cas.
</div>

```python
# On peut également supperposer l'histogramme et le kde plot avec seaborn `histplot`
# (kde=True, binds='fd', edgecolor=None)
...
```

<div class="alert alert-block alert-info">
<b>L'axe des y sur un kde plot - partie 2</b>
<br> Lorsque l'on supperpose un histogramme avec un kde avec `seaborn`, vous remarquerez que l'axe des y montre la fréquence. Cependant, la courbe kde reste tout de même une courbe de densité. Cela se produit car seaborn met à l'échelle la courbe densité avec l'histogramme en multipliant la courbe par le nombre d'observation et la taille des bins. 
</div>


#### Nuage de points univarié

L'**histogramme** et l'**estimation de la densité par noyau** nous permettent de visualiser la distribution d'une variable de manière discrète ou continue, respectivement. Cependant, ces types de visualisations ne nous permettent pas de visualiser les données brutes.

Le **nuage de points univarié** nous permet de visualiser chaque point de données individuellement. Cela peut nous permettre d'identifier plus facilement la présence de valeurs abbérantes dans nos données. Cependant, le nuage de points n'est pas adapté si nous avons trop de points de données.

```python
# Utiliser la fonction `stripplot` de la librairie seaborn
...
```

<div class="alert alert-block alert-warning">
<b>Réplication du nuage de points</b>
<br>Essayez de répliquer le nuage de points de la variable `Age` dans deux cellules différentes ci-dessous. Que remarquez-vous ?
</div>

```python
# À compléter
```

```python
# À compléter
```

<div class="alert alert-block alert-info">
<b>Le paramètre `jitter`</b>
<br>Vous avez peut-être remarqué que les deux nuages de points que vous avez générés à partir de la même variable ne sont pas tout à fait identiques. Cela se produit car `seaborn` utilise `numpy.random` pour le calcul du <i>jitter</i>. Pour rendre le calcul du <i>jitter</i> reproductible, il est possible de fixer une <i>seed</i> a priori avec `np.random.seed()`.
</div>

```python
...
```

```python
...
```

#### Diagramme à barres

Le **diagramme à barres** (*bar plot*) permet de visualiser/comparer des variables catégorielles en montrant les fréquences des différentes valeurs ou simplement les différentes valeurs. Ce type de graphique est utile si l'on veut, par exemple, comparer le nombre de personnes par groupe.

```python
participants['AgeGroup'].value_counts()
```

```python
participants['AgeGroup'].value_counts().plot(kind='bar')
plt.ylabel('Counts')
```

```python
order = sorted(participants.AgeGroup.unique())

sns.countplot(
    data=participants,
    x='AgeGroup',
    order=order,
    hue='Gender',
)
```

### Résumé

| Type de graphique | Type de données | Résumé |
| --- | --- | --- |
| Histogramme | Continue | Pour visualiser la distribution de nos données de manières discrètes |
| Estimation de la densité par noyau | Continue | Pour visualiser la distribution de nos données de manières continues |
| Nuage de points | Continue | Pour visualiser chaque point de données individuellement |
| Graphique à barres | Catégorielle | Pour comparer les fréquences entre différents groupes/catégories |



### Visualisations bivariées

Pour une **variable continue** x **variable continue**, nous pouvons utiliser:
- Nuage de points
- Estimation de la densité par noyau bivariée
- *Hexplot*
- *Joint plot*
- *Heat map*

Pour une **variable catégorielle** x **variable continue**, nous pouvons utiliser:
- Boîte à moustache (*box plot*)
- Diagramme en violon (*violin plot*)
- Nuage de points (*scatter plot*)
- Tracé de points (*point plot*)
- *Rain cloud plot*
- Diagramme à barres (*bar plot*)... (?)


#### Nuage de points - Variable continue x variable continue

```python
# Regardons la relation entre 'Age' et 'ToM Booklet-Matched' avec la fonction matplotlib `scatter`
...
```

<div class="alert alert-block alert-warning">
<b>Que remarquez-vous ?</b>
<br>Prenez le temps d'observer la figure que nous venons de générer.
</div>

```python
participants.groupby(['Child_Adult'])['ToM Booklet-Matched'].mean()
```

<div class="alert alert-block alert-info">
<b>Valeurs manquantes</b>
<br>La fonction `scatter` dans `matplotlib`, ainsi que la fonction `regplot` dans `seaborn` (voir les cellules suivantes), permettent de supprimer automatiquement les valeurs manquantes lors de la visualisation.
</div>


<div class="alert alert-block alert-info">
<b>La fonction `regplot`</b>
<br>La fonction `regplot` dans seaborn permet de visualiser le nuage de points tout en ajustant un modèle de régression linéaire aux données !
</div>

```python
# Regardons la relation entre 'Age' et 'ToM Booklet-Matched' avec la fonction `regplot` de seaborn
...
```

<div class="alert alert-block alert-warning">
<b>Ajustement du modèle de régression</b>
<br>Une régression linéaire ne semble pas être la meilleure façon de modéliser la relation entre notre variable `Age` et notre variable `ToM Booklet-Matched`. Modifier la valeur du paramètre `order` de la fonction `regplot` à 2 pour vérifier l'ajustement de ce modèle. 
</div>

```python
# À compléter
```

<div class="alert alert-block alert-info">
<b>Et si nous ajoutions une variable de plus !</b>
<br>Nous pouvons visualiser les interactions multivariées grâce au paramètre `hue` de la fonction `lmplot` de `seaborn`. Dans la cellule ci-dessous, nous allons explorer la relation entre l'âge (<b>x</b>) et le score au <i>ToM Booklet-Matched</i> (<b>y</b>) en fonction du genre (<b>hue</b>).
</div>

```python
# Regardons la relation entre 'Age', 'ToM Booklet-Matched' et 'Gender' avec la fonction `lmplot` de seaborn
...
```

#### Estimation de la densité par noyau bivariée et hex plot - Variable continue x variable continue

Lorsque nous avons beaucoup de points de données et que nous voulons visualiser la distribution de points en prenant en compte deux variables, nous risquons d'avoir beaucoup de chevauchement de points (*overplotting*). Dans ce cas, il peut être difficile de bien visualiser la distribution de nos données avec un nuage de points. Il est donc possible d'utiliser d'autres types de graphiques:

L'**estimation de la densité par noyau bivariée** nous permet de visualiser la manière dont deux variables se distribuent dans un espace à deux dimensions. Chaque contour représente une zone densité. Plus les contours sont proches, plus la densité est élevée, c'est-à-dire là où les données sont le plus concentrées. 

Le **hex plot** permet de visualiser la densité des points de données de manière discrète. C'est donc l'équivalent d'un histogramme mais pour visualiser deux variables plutôt qu'une. Les zones plus foncées représentent des zones de densité élévée.

&#x26a0; L'estimation de la densité par noyau bivariée est plus demandant en termes de computation comparativement au *hex plot*.

```python
sns.kdeplot(
    x=participants['Age'], 
    y=participants['ToM Booklet-Matched'],
    fill=True
)
```

```python
sns.jointplot(
    x=participants['Age'], 
    y=participants['ToM Booklet-Matched'],
    kind='hex'
)
```

```python
def plot_jointplot(kind):
    sns.jointplot(
        x=participants['Age'], 
        y=participants['ToM Booklet-Matched'],
        kind=kind
    )
    plt.show()

widgets.interact(
    plot_jointplot,
    kind=widgets.Dropdown(
        options=['hex', 'reg', 'kde', 'scatter'],
        description='Type:'
    )
);
```

```python
def plot_jointplot(kind):
    mean, cov = [0, 1], [(1, .5), (.5, 1)]
    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    
    sns.jointplot(
        x=x, 
        y=y,
        kind=kind
    )
    plt.show()

widgets.interact(
    plot_jointplot,
    kind=widgets.Dropdown(
        options=['scatter', 'hex', 'reg', 'kde'],
        description='Type:'
    )
);
```

#### Nuage de points (stripplot) - Variable continue x variable catégorielle

Nous avons déjà parlé du nuage de points (*stripplot*) au niveau univarié, mais nous pouvons également visualiser le nuage de points de plusieurs catégories simultanément.

```python
order = sorted(participants.AgeGroup.unique())[:-1]

sns.stripplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'], 
    order=order
)
```

#### Boîtes à moustache - Variable continue x variable catégorielle

Les **boîtes à moustaches** nous permettent de visualiser les distributions d'un ou plusieurs groupes de variables continues (p.ex. distributions d'âge pour différents groupes expérimentaux). Les différentes composantes de la boîte à moustache représentent différentes statistiques descriptives:
- La ligne dans la boîte représente la médiane.
- Les limites de la boîte (quartiles inférieur—Q1 et supérieur—Q3) représentent l'étendue où se situe 50% des données.
- Les moustaches (c.-à-d. les lignes à l'extérieur de la boîte) capturent l'étendue du reste des données.
- Les points représentent les valeurs aberrantes—valeurs supérieures à 1.5 fois l'intervalle inter-quartile (Q3-Q1) + Q3 ou inférieures à 1.5 fois l'intervalle inter-quartile - Q1.

```python
sns.boxplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'],
    order=order
)
```

#### Diagrammes en violon - Variable continue x variable catégorielle

Les **diagrammes en violon** nous permettent de visualiser la distribution des données en utilisant des courbes de densité (aka courbes d'**estimation de la densité par noyau**). La largeur de chaque courbe correspond à la fréquence approximative des points pour chaque région (valeurs sur l'axe des y).

```python
sns.violinplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'], 
    order=order
)
```

#### Diagrammes raincloud - Variable continue x variable catégorielle

Pourquoi choisir entre un nuage de points, une boîte à moustaches et un diagramme en violon quand nous pouvons faire les trois en même temps ! C'est ce que nous permettent de faire les **diagrammes raincloud** (*raincloud plot*).

*Note :* Ce type de graphique n'est pas nativement intégré par `seaborn` ou `matplotlib`. Nous devrons utiliser la librairie `ptitprince` que nous avons importé au début du tutoriel (`import ptitprince as pt`).

*Ressource :* Pour plus d'exemples sur les RainCloud plot, voir le tutoriel de [ptitprince](https://github.com/pog87/PtitPrince/blob/master/tutorial_python/raincloud_tutorial_python.ipynb).

```python
pt.RainCloud(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'], 
    order=order,
    bw=0.6
)
```

#### Tracé de points - Variable continue x variable catégorielle

Le **tracé de points** nous permettent de comparer les moyennes (ou autre statistique descriptive) entre différents groupes, tout en montrant l'incertitude (p. ex. intervalle de confiance à 95%, écart-type, etc.). Ce type de visualisation est pratique pour montrer des tendances entre différentes catégories ou entre différents temps de mesure (p.ex. si nous avons des mesures longitudinales).

```python
sns.pointplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    estimator='mean',
    data = participants[participants.AgeGroup!='Adult'], 
    order=order,
    errorbar=('ci', 95)
)
```

#### Diagramme à barres pour visualiser une variable continue x une variable catégorielle ?

Vous avez peut-être déjà vu l'utilisation de diagramme à barres pour représenté un variable continue.
Cependant, ce type de visualisation pour ce type de variable n'est pas recommandé:
- Permet uniquement de visualiser certains statistiques descriptives (p.ex. la moyenne) sans fournir aucune information sur la distribution de notre variable
- Si vous voulez absolument utiliser un diagramme à barres, superposez-le avec un nuage de points !

```python
sns.barplot(
    x='AgeGroup',
    y = 'ToM Booklet-Matched',
    data = participants[participants.AgeGroup!='Adult'],
    order = order, palette='Blues'
)


sns.stripplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched',
    data = participants[participants.AgeGroup!='Adult'],
    jitter=True,
    order = order, color = 'black')

```

## Et si on ajoutait un peu de couleurs à nos figures ?


### Palettes de couleurs perceptuelle uniformes vs non-uniformes
- Couleurs sont perçues selon leur teinte (orange, rouge, vert, etc.) et leur luminosité (clareté vs obscurité d'une teinte)
- Les caractéristiques de nos photorécepteurs font en sorte que nous ne traitons pas le spectre de lumière de manière uniforme
- La majorité des photorécepteurs nous permettant de voir les couleurs (cônes) traite les longueurs d'onde longues (c.-à-d., rouge, orange, jaune)
- Donc, nous ne percevons pas très bien les variations dans les teintes vertes-bleues comparativement aux variations dans les teintes jaunes-rouges

```python
colormaps['hsv']
```

```python
cm.batlow
```

```python
import requests
from PIL import Image
from io import BytesIO
```

```python
response = requests.get('https://raw.githubusercontent.com/matplotlib/matplotlib/main/doc/_static/stinkbug.png')
img = np.asarray(Image.open(BytesIO(response.content)))
```

```python
def plot_img(cmap):
    lum_img = img[:, :, 0]
    if cmap == 'noir et blanc':
        plt.imshow(img)
    elif cmap == 'hsv':
        plt.imshow(lum_img, cmap="hsv")
    elif cmap == 'batlow':
        plt.imshow(lum_img, cmap=cm.batlow)

widgets.interact(
    plot_img,
    cmap=widgets.Dropdown(
        options=['noir et blanc', 'hsv', 'batlow'],
        value='noir et blanc',
        description='Palette:'
    )
);
```

<div class="alert alert-block alert-warning">
<b>Que remarquez-vous ?</b>
<br>Comparer l'image colorée avec les palettes de couleurs `hsv` et `batlow` avec l'image en noir et blanc.
</div>


<div class="alert alert-block alert-info">
<b>Ne jetez pas l'arc-en-ciel à la poubelle</b>
<br> Google a développé la palette de couleur `turbo`, une palette de couleur arc-en-ciel perceptuellement uniforme. Pour plus de détails, consultez <a href="https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/">le blog de Google research</a>.
</div>


### Plattes de couleurs discrètes vs continues

Les palettes de couleurs peuvents soient être continues (comme ce que nous avons vu plus haut) ou discrète. Les palettes de couleurs discrètes sont utilisées pour visualiser les variables catégorielles—où les catégories n'ont pas d'ordre inhérent (p. ex. Enfants avec covid vs enfants sans covid vs adultes avec covid vs adultes sans covid).


#### [matplotlib](https://matplotlib.org/stable/gallery/color/colormap_reference.html)

```python
colormaps['Set2']
```

```python
colormaps['Set3']
```

#### [seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html)

```python
sns.color_palette('rocket', 10)
```

#### [cmcrameri](https://s-ink.org/scientific-colour-maps)

```python
cm.batlow.resampled(10)
```

```python
cm.lipari.resampled(10)
```

<div class="alert alert-block alert-info">
<b>Discrétisation des palettes continues</b>
<br>Dans les cellules ci-dessous, nous avons vu qu'il était possible de discrétiser une palette continue en spécifiant le nombre de couleurs que nous voulons. Par contre, comme nous l'avons mentionné les palettes discrètes sont utilisées pour visualiser des catégories qui n'ont pas d'ordre inhérent. Donc nous ne sommes pas obligé d'avoir une palette discrète ordonnée.
</div>

```python
nb_colors = 10
#np.random.seed(10)

original_cmap = cm.lipari.resampled(nb_colors)

original_colors = original_cmap(np.arange(nb_colors))

shuffled_colors = original_colors.copy()
np.random.shuffle(shuffled_colors)

colors.ListedColormap(shuffled_colors)
```

### Plattes de couleurs divergentes

Les palettes de couleurs divergentes sont utiles si nous avons une valeur centrale interprétable. Les palettes divergentes s'appliquent autant pour les palettes discrètes ou continues. Par exemple:
- **Palettes discrètes divergentes**: Si nous avons des données collectées grâce à une échelle de likert. Nos valeurs pourraient varier de 'Fortement en désaccord' à 'Fortement en accord' avec comme valeur centrale 'Neutre'. Dans ce cas, nous pourrions visualiser les valeurs de gauche (de 'Fortement en désaccord' à 'neutre') dans des teintes de bleues et les valeurs de droite (de 'Neutre' à 'Fortement en accord') dans les teintes de oranges/rouges.
- **Palettes continues divergentes**: Si nous avons des valeurs négatives et positives (p.ex. des valeurs de corrélations), nous pourrions utiliser le zéro comme valeur centrale, les valeurs négatives pourraient être représentées dans des teintes de bleues et la valeurs positives dans des teintes de oranges/rouges.


#### [matplotlib](https://matplotlib.org/stable/gallery/color/colormap_reference.html)

```python
# Palette discrète divergente
colormaps['bwr'].resampled(9)
```

```python
# Palette continue divergente
colormaps['bwr']
```

#### [seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html)

```python
sns.color_palette("coolwarm", 9)
```

```python
sns.color_palette("coolwarm", as_cmap=True)
```

#### [cmcrameri](https://s-ink.org/scientific-colour-maps)

```python
cm.vik.resampled(9)
```

```python
cm.vik
```

### Palettes de couleurs lisibles universellement

Nous avons parlé de l'uniformité perceptuelle des palettes de couleurs, mais il est également important de considérer l'utilisation de couleurs perçues universellement. En effet, l'utilisation de certaines combinaisons de couleurs pourrait être difficile à distinguer pour des personnes ayant une dyschromatopsie. Quelques conseils:
- Évitez les combinaisons de vert et rouge
- Variez la luminosité et la saturation des couleurs pour s'assurer que vos figures restent lisibles en noir et blanc
- Utilisez les palettes de couleur lisibles universellement de `matplotlib`, `seaborn` ou `cmcrameri`


```python
sns.color_palette("colorblind")
```

```python
converters = {
    'deuter50_space': {
        "name": "sRGB1+CVD",
        "cvd_type": "deuteranomaly",
        "severity": 50
    },
    'deuter100_space': {
        "name": "sRGB1+CVD",
        "cvd_type": "deuteranomaly",
        "severity": 100
    },
    'prot50_space': {
        "name": "sRGB1+CVD",
        "cvd_type": "protanomaly",
        "severity": 50
    },
    'prot100_space': {
        "name": "sRGB1+CVD",
        "cvd_type": "protanomaly",
        "severity": 100
    },
    'trit50_space': {
        "name": "sRGB1+CVD",
        "cvd_type": "tritanomaly",
        "severity": 50
    },
    'trit100_space': {
        "name": "sRGB1+CVD",
        "cvd_type": "tritanomaly",
        "severity": 100
    }
}

def show_palettes(cmap, n_colors=10):
    f, ax = plt.subplots(7, 1, figsize=(10, 6)) #, layout="constrained")
    plt.subplots_adjust(hspace=0.6)
    
    gradient = np.arange(n_colors).reshape(1, -1)

    if cmap in ['viridis', 'coolwarm', 'colorblind' ,'pastel', 'muted']:
        cmap_sns = sns.color_palette(cmap, n_colors)
        cmap_m = colors.ListedColormap(sns.color_palette(cmap, n_colors=n_colors))
    if cmap in ['batlow', 'vik']:
        cmap_m = getattr(cm, cmap).resampled(n_colors)
        cmap_sns = sns.color_palette(cmap_m(np.linspace(0, 1, n_colors)))
    if cmap in ['bwr', 'hsv', 'PuOr', 'summer']:
        cmap_m = colormaps[cmap].resampled(n_colors)
        cmap_sns = sns.color_palette(cmap_m(np.linspace(0, 1, n_colors)))

    # Original palette
    ax[0].imshow(gradient, aspect='auto', cmap=cmap_m)
    ax[0].set_title('Original')
    for idx, converter in enumerate(converters.keys()):
        # Palette transformée
        converted_palette = cspace_convert(cmap_sns, converters[converter], "sRGB1")
        converted_palette = np.clip(converted_palette, 0, 1)

        ax[idx+1].imshow(gradient, aspect='auto', cmap=colors.ListedColormap(converted_palette))
        ax[idx+1].set_title(f"{converters[converter]['cvd_type']}-{converters[converter]['severity']}")

    for a in ax.flatten():
        a.set_yticks([])
        a.set_xticks([])
        
    plt.show()

widgets.interact(
    show_palettes,
    cmap=widgets.Dropdown(
        options=['pastel', 'viridis', 'coolwarm', 'batlow', 'bwr', 'colorblind', 'hsv', 'PuOr', 'summer', 'vik', 'muted'],
        value='viridis',
        description='Palette:'
    ),
);

```

<div class="alert alert-block alert-info">
<b>Au-delà des couleurs</b>
<br> Choisir la bonne palette de couleurs est important, mais il existe également d'autres stratégies que nous pouvons utiliser pour rendre nos figures plus accessibles et facilement interprétables. Au lieu d'utiliser uniquement différentes couleurs pour distinguer différentes catégories, nous pourrions utiliser <b>différentes formes géométriques</b>. Si notre figure comporte des lignes pour, par exemple, illustrer la trajectoire d'une variable dans le temps, nous pourrions utiliser <b>différents types de pointillés</b>. Pour plus d'exemples, voir <a href="https://www.datylon.com/blog/data-visualization-for-colorblind-readers">"The best charts for color blind viewers"</a>.
</div>


## L'anatomie d'une figure

Nous avons déjà vu comment ajouter ou modifier quelques éléments de nos figures, comme le titre et le nom des axes. Dans cette section, nous allons discuter plus en détails de l'art de faire des figures. Nous allons couvrir:
- Sous-graphes (*subplots*)
- Bordures
- Graduations (*ticks*)
- Grille
- Légende


### Sous-graphes

Jusqu'à présent nous avons principalement créé nos figures en appelant directement certaines fonctions de `matplotlib` et de `seaborn`. Par contre, si nous voulons créer une figure avec plusieurs panneaux/axes (c.-à-d., colonnes et/ou rangées), nous allons devoir utiliser des **sous-graphes** (*subplots*).


#### matplotlib

```python
# Reprenons le code que nous avons utilisé précédemment. Ce code produit deux figures séparées
plt.hist(participants['Age'], bins='fd')
plt.title("Distribution de l'age")
plt.xlabel('Age')
plt.ylabel('Compte')
plt.show()

plt.scatter(participants['Age'], participants['ToM Booklet-Matched'])
plt.xlabel('Age')
plt.ylabel('ToM Booklet-Matched')
plt.show()
```

```python
help(plt.subplots)
```

```python
# Si nous voulons produire une seule figure avec ces deux graphiques nous allons devoir utiliser les subplots
# Syntaxe de la fonctionL plt.subplots(n_rows, n_cols, *)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(participants['Age'], bins='fd')
axes[0].set_title("Distribution de l'age")
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Compte')

axes[1].scatter(participants['Age'], participants['ToM Booklet-Matched'])
axes[1].set_title("Relation entre Age et scores ToM")
axes[1].set_xlabel('Age')
axes[1].set_ylabel('ToM Booklet-Matched')
```

```python
# La fonction subplots peut même être utilisé lorsque l'on a un seul graphe
fig, axes = plt.subplots(figsize=(6, 4))
axes.hist(participants['Age'], bins='fd')
axes.set_title("Distribution de l'age")
axes.set_xlabel('Age')
axes.set_ylabel('Compte')
```

#### seaborn

```python
# Avec searbon
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

order = sorted(participants[participants.AgeGroup!='Adult']['AgeGroup'].unique())
#order.sort()

sns.boxplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'],
    order=order,
    ax=axes[0]
)

sns.violinplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'],
    order=order,
    ax=axes[1]
)
```

### Bordure
La bordure correspond aux lignes autours de la zone de tracé.

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Distribution de l'age")
ax.set_xlabel('Age')
ax.set_ylabel('Compte')

for spine in ax.spines.values():
    spine.set_color("red")
    spine.set_linewidth(3)
```

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Distribution de l'age")
ax.set_xlabel('Age')
ax.set_ylabel('Compte')

ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False) # ax.spines.top.set_visible(False)
```

### Graduations

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Distribution de l'age")
ax.set_xlabel('Age')
ax.set_ylabel('Compte')

ax.tick_params(
    axis='both', # Modification appliquée aux deux axes (x et y)
    which='major', # Modification des graduations 'major', 'minor' ou 'both'
    length=10, # Longueur des graduations
    width=2, # largeur des graduations
    color='purple', # Couleur des graduations
    labelsize=12, # Taille des étiquettes
    labelcolor='darkorange', # Couleur des étiquettes
    labelrotation=45
)

#from matplotlib.ticker import MultipleLocator
#ax.xaxis.set_minor_locator(MultipleLocator(2))
```

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Distribution de l'age")
ax.set_xlabel('Age')
ax.set_ylabel('Compte')

plt.tick_params(
    axis='x',
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
```

<div class="alert alert-block alert-warning">
<b>Modifiez le code</b>
<br> À partir du code fourni dans les cellules ci-dessus, modifiez la cellule du bas pour générer une figure avec les caractéristiques suivantes:
<li>
    Pas de bordure au haut et à gauche
</li>
<li>
    Pas de graduation sur l'axe des y
</li>
<li>
    Des graduations mineurs sur l'axe des x à intervalle de 1
</li>
<li>
    Les étiquettes sur l'axe des x affichées à 90 degré
</li>
<li>
    Des titres pour tous les axes ainsi que pour la figure
</li>

```python
# Ajouter votre code dans cette cellule
# ...
```

### Grille

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Distribution de l'age")
ax.set_xlabel('Age')
ax.set_ylabel('Compte')


ax.grid(
    True,
    color='gray',
    linestyle='--',
    linewidth=1
)

ax.set_axisbelow(True) # Équivalent à l'utilisation du paramètre `zorder`
```

### Légende

```python
fig, ax = plt.subplots(figsize=(6, 4))

sns.scatterplot(data=participants, x='Age', y='ToM Booklet-Matched', hue='Gender', ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('ToM Booklet-Matched')

legend = ax.legend(fontsize=12, loc='lower right')

legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(2)
legend.get_frame().set_facecolor("white")
```

### Paramètres par défaut

Les paramètres par défaut pour tous les éléments de nos figures (p. ex. police, taille du texte, épaisseur des lignes, etc.) sont définis dans l'object `rcParams`. Ces valeurs peuvent être modifiées, nous permettant d'appliquer un style de figure consistant d'une figure à l'autre.

```python
plt.rcParams
```

```python
STYLE = {
    "font.cursive": "Comic Sans MS",
    "font.size": 8,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "axes.grid.axis": 'both',
    'axes.axisbelow': True,
    "figure.dpi": 150
}

plt.rcParams.update(STYLE)
```

```python
fig, axes = plt.subplots(figsize=(6, 4))
axes.hist(participants['Age'], bins='fd')
axes.set_title("Distribution de l'age")
axes.set_xlabel('Age')
axes.set_ylabel('Compte')
```

```python
# Pour remettre les valeurs par défaut
plt.rcdefaults()
```

<div class="alert alert-block alert-info">
<b>Choisissez votre style</b>
<br>Il est également possible d'utiliser des styles pré-définis grâce à la fonction `style.use` de matplotlib. Vous pouvez consulter la liste de styles offerts <a href=https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>ici</a>. Voir également <a href='https://matplotlib.org/stable/users/explain/customizing.html'>la documentation de matplotlib</a> pour en apprendre plus sur les feuilles de style et `rcParams`.
</div>

```python
print(plt.style.available)
```

<div class="alert alert-block alert-info">
Par exemple, 'ggplot' est un style que nous pouvons utiliser pour obtenir des figures semblables à celles générées par la librairie `ggplot` en R.
</div>

```python
plt.style.use('ggplot')
```

```python
fig, axes = plt.subplots(figsize=(6, 4))
axes.hist(participants['Age'], bins='fd')
axes.set_title("Distribution de l'age")
axes.set_xlabel('Age')
axes.set_ylabel('Compte')
```

## Graphiques interactifs

Les figures statiques c'est bien, mais les figures interactives c'est mieux ! En fait, les graphiques interactifs nous offrent l'opportunité d'explorer nos données d'une manière qui ne serait pas possible avec des figures statiques et de créer des dashboards (voir [documentation](https://dash.plotly.com/?_gl=1*j1jto0*_gcl_au*MTY3MTkxNTc4MS4xNzczOTMwNTAw*_ga*MTM4NzMyMTAxMy4xNzczOTMwNTAy*_ga_6G7EE0JNSC*czE3NzM5MzA1MDIkbzEkZzAkdDE3NzM5MzA1MTAkajUyJGwwJGgw)). Nous allons pouvoir ajouter des informations à nos figures, sans les encombrer.

Les deux principales librairies en python nous permettant de faire des figures interactives sont `plotly` et `bokeh`. La librairie `plotly` offre un interface de haut niveau (`plotly.express`) permettant de créer des figures en une seule ligne de code, alors que `bokeh` offre plus d'options de personnalisation. Dans ce tutoriel, nous allons uniquement discuter de `plotly.express`, mais si vous voulez essayer `bokeh`, vous pouvez consulter [la documentation](https://docs.bokeh.org/en/latest/).

```python
help(px.scatter)
```

```python
fig = px.scatter(
    data_frame=participants, 
    x='Age', 
    y='ToM Booklet-Matched',
    hover_data=['participant_id', 'Gender'],
    color='Handedness',
    symbol='Handedness'
)

fig.show()
```

```python
participants.columns
```

```python
fig = px.scatter(
    data_frame=participants, 
    x='Age', 
    y='ToM Booklet-Matched',
    hover_data=['participant_id', 'Gender'],
    color='Handedness',
    symbol='Handedness'
)
fig.update_xaxes(range=[int(participants['Age'].min()-1), int(participants[participants['Child_Adult']=='child']['Age'].max()+1)])
fig.update_traces(marker_size=10)
fig.show()
```

## Visualiser des images de cerveaux !

La librairie `nilearn` supporte plusieurs fonctions pour visualiser des images de cerveaux (voir la [liste de toutes les fonctions](https://nilearn.github.io/stable/plotting/index.html)). Dans cette section, nous allons voir quelque unes de ces fonctions.

---

Basé sur le [tutoriel MAIN](https://main-educational.github.io/intro_nilearn/machine-learning-with-nilearn.html)

```python
# Allons tout d'abord chercher nos données
data = development_dataset.func
confounds = development_dataset.confounds
pheno = pd.DataFrame(development_dataset.phenotypic)
```

```python
data
```

```python
# Essayons de visualiser notre premier fichier
plotting.view_img(data[0])
```

```python
img = nib.load(data[0])
img.shape
```

<div class="alert alert-block alert-info">
<b>Oups !</b>
<br> Si nous essayons de visualiser nos images bold, nous obtenons une erreur ! C'est tout à fait normal, puisque notre fichier contient des données 4D, c'est-à-dire que pour chaque voxel (3D), nous avons une valeur pour plusieurs points dans le temps (temps de répétition; +1D). Si nous voulons absoluement visualiser ces fichiers, deux options s'offrent à nous:
<ul>
    1. Soit nous nous intéressons à l'activité sur l'ensemble du cerveau à un temps donné
</ul>
<ul>
    2. Soit nous nous intéressons au décours temporel pour un voxel/parcelle donné
</ul>   
</div>

```python
# Allons chercher notre premier volume (i.e., notre première image de cerveau)
premier_volume = image.index_img(data[0], 0)

plotting.view_img(premier_volume, black_bg=False, cmap='turbo', symmetric_cmap=False)
```

```python
help(plotting.plot_stat_map)
```

```python
plotting.plot_stat_map(
    premier_volume, 
    draw_cross=False,
    #cut_coords=(0, 4, 22),
    display_mode='tiled'
)
```

```python
multiscale = datasets.fetch_atlas_basc_multiscale_2015(resolution=64, data_dir='../data')
atlas_filename = multiscale.maps

# initialize masker (change verbosity)
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', resampling_target="data",
                           detrend=True, verbose=0)
# Extraction des séries temporelles pour notre premier sujet
time_series = masker.fit_transform(data[0], confounds=confounds[0])
```

```python
time_series.shape
```

```python
parcelle = 0
plt.figure(figsize=(12,4))
plt.plot(time_series.T[parcelle])
plt.title(f'Décours temporel de la parcelle {parcelle}')
plt.xlabel('Volumes')
plt.ylabel('Amplitude')
```

<div class="alert alert-block alert-warning">
<b>Comparer visuellement les décours temporels</b>
<br>À partir du code fourni dans la cellule d'avant, modifier la cellule ci-dessous pour être en mesure de comparer le décours temporel de la <b>parcelle 0</b> et de la <b>parcelle 1</b>.
</div>

```python
help(plt.legend)
```

```python
#À compléter
```

### Charger les données

```python
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True,
                                         discard_diagonal=True)


all_features = [] # here is where we will put the data (a container)

for i,sub in enumerate(data[:66]):
    # extract the timeseries from the ROIs in the atlas
    time_series = masker.fit_transform(sub, confounds=confounds[i])
    # create a region x region correlation matrix
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    # add to our container
    all_features.append(correlation_matrix)
    # keep track of status
    print('finished %s of %s'%(i+1,len(data[:66])))

np.savez_compressed('data/MAIN_BASC064_subsamp_features', a=all_features)
```

<div class="alert alert-block alert-danger">
Si vos données ne se trouvent pas dans le dossier <i>data/</i>, modifier le chemin dans la cellule ci-dessous.
</div>

```python
y_ageclass = pheno.head(66)['Child_Adult']

feat_file = 'data/MAIN_BASC064_subsamp_features.npz'
X_features = np.load(feat_file)['a']
```

```python
print(f"Shape X: {X_features.shape}")
print(f"Shape y: {y_ageclass.shape}")
```

```python
sns.countplot(x = y_ageclass)
```

### Entraîner le modèle

```python
from sklearn.model_selection import train_test_split

# Split the sample to training/test and
# stratify by age class, and also shuffle the data.

X_train, X_test, y_train, y_test = train_test_split(X_features, # x
                                                    y_ageclass, # y
                                                    test_size = 0.2, # 80%/20% split  
                                                    shuffle = True, # shuffle dataset
                                                                    # before splitting
                                                    stratify = y_ageclass, # keep
                                                                           # distribution
                                                                           # of ageclass
                                                                           # consistent
                                                                           # betw. train
                                                                           # & test sets.
                                                    random_state = 123 # same shuffle each
                                                                       # time
                                                    )

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

scaler = StandardScaler().fit(X_train)
X_train_scl = scaler.transform(X_train)
X_test_scl = scaler.transform(X_test)

l_svc = SVC(kernel='linear', class_weight='balanced')

l_svc.fit(X_train_scl, y_train) # fit to training data
y_pred = l_svc.predict(X_test_scl) # classify age class using testing data

acc = l_svc.score(X_test_scl, y_test) # get accuracy
cr = classification_report(y_pred=y_pred, y_true=y_test) # get prec., recall & f1
cm = confusion_matrix(y_pred=y_pred, y_true=y_test) # get confusion matrix
```

```python
print(cr)
```

### Visualisation des coefficients

Nous avons entraîné notre modèle et obtenu un score de prédiction assez élevé, ce qui nous indique qu'il y a possiblement quelque chose dans nos données qui est systématiquement lié à l'âge. 

```python
print(l_svc.coef_.shape)
print(l_svc.coef_)
```

#### Matrice de corrélation
Les attributs de notre modèle correspond à la corrélation entre chaque paire de régions que nous avons extraites. Les coefficients de notre modèle représente donc le poids de chacune de ces paires de régions dans la prédiction du groupe d'âge. Nous pouvons donc uiliser une matrice de corrélation pour visualiser ces poids.

```python
feat_exp_matrix = correlation_measure.inverse_transform(l_svc.coef_)[0]

plotting.plot_matrix(feat_exp_matrix, figure=(10, 8),  
                     labels=range(feat_exp_matrix.shape[0]),
                     reorder='average',
                    tri='lower', vmax=0.01, vmin=-0.01)
```

#### Connectome

Nous pouvons aussi visualiser directement le poids de nos attributs sur un cerveau !

```python
# Coordonnées de nos régions
coords = plotting.find_parcellation_cut_coords(atlas_filename)
print(coords.shape)
```

```python
plotting.plot_connectome(feat_exp_matrix, coords, colorbar=True)
```

```python
plotting.plot_connectome(feat_exp_matrix, coords, colorbar=True, edge_threshold=0.006)
```

<div class="alert alert-block alert-info">
<b>Ajoutons du mouvement !</b>
<br>Nilearn possède une fonction nous permettant de visualiser notre connectome de manière interactive ! Cela nous permet de plus facilement examiner le poids de nos attributs.
</div>

```python
plotting.view_connectome(feat_exp_matrix, coords, edge_threshold='90%')
```

<div class="alert alert-block alert-warning">
<b>Attributs tous gris...</b>
<br>Vous avez peut-être remarqué que les poids de nos attributs sont tracés en gris. Pourquoi pensez-vous que c'est le cas ?
</div>

```python
feat_exp_matrix_rm_diag = feat_exp_matrix
feat_exp_matrix_rm_diag[feat_exp_matrix==1] = 0
plotting.view_connectome(feat_exp_matrix_rm_diag, coords, edge_threshold='90%')
```

Nous avons un modèle permettant de prédire le groupe d'âge avec une très bonne performance prédictive. Nous pouvons voir que les attributs nous permettant de faire cette prédiction sont distribués dans le cerveau. Est-ce que nous pouvons maintenant publier nos résultats ?...
<br>
<br>Non ! Il nous faut explorer davantage pour voir si notre modèle est biologiquement plausible... Pour cela, nous allons visualiser nos images cérébrales pour chacun de nos groupes.

```python
children, adults = data[33:66], data[0:33]
avg_children, avg_adults = [], []

# Pour chaque participant.e, nous allons moyenner l'activité cérébrale à travers tous nos points de mesure pour obtenir une image 3D
for child, adult in zip(children, adults):
    avg_adults.append(image.mean_img(adult))
    avg_children.append(image.mean_img(child))

# Nous allons moyenner nos images 3D individuelles pour chaque participant.e et ce pour chacun de nos groupes séparément
avg_children = image.mean_img(avg_children)
avg_adults = image.mean_img(avg_adults)
```

```python
plotting.view_img(avg_children, black_bg=False, cut_coords=(0,-16,16))
```

```python
plotting.view_img(avg_adults, black_bg=False)
```

<div class="alert alert-block alert-warning">
<b>Que remarquez-vous ?</b>
<br>Regardez les images moyennées pour chacun des groupes. Pouvez-vous observer certaines différences ?
</div>

```python
# Allons chercher notre premier volume pour notre premier participant
premier_volume = image.index_img(data[0], 0)

plotting.view_img(premier_volume, black_bg=False, cmap='turbo', symmetric_cmap=False)
```

```python
# Allons chercher notre premier volume pour notre 39e participant
premier_volume = image.index_img(data[40], 0)

plotting.view_img(premier_volume, black_bg=False, cmap='turbo', symmetric_cmap=False)
```

## Ressources supplémentaires

- [Tutoriel de `seaborn` sur la visualisation des distributions de données](https://seaborn.pydata.org/tutorial/distributions.html)
- [Python Graph Gallery](https://python-graph-gallery.com/)
- [Guide compréhensif de la visualisation](https://www.atlassian.com/data/charts)
- [Guide des pratiques de visualisation à éviter](https://www.data-to-viz.com/caveats.html)
- [Fonctions de visualisation dans nilearn - Données IRM(f)](https://nilearn.github.io/dev/plotting/index.html)
- [Tutoriels de visualisation dans MNE python - Données EEG/MEG](https://mne.tools/stable/auto_tutorials/visualization/index.html)
- [Tutoriels de visualisation dans DIPY - Données IRM de diffusion](https://docs.dipy.org/stable/examples_built/index#visualization)
