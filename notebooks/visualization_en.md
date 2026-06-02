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
# Visualization and Model Interpretation

## Learning Objectives
&#x1F440; Understand the purpose of visualization
<br> &#x1F4C8; Understand which type of graph to use according to the data type
<br> &#x1F3A8; Adequately choose your color palette
<br> &#x1F503; Learn how to modify different elements of our figures
<br> &#x1f9e0; Use `nilearn` for neuroimaging data visualization

## Tutorial Organization

The session will consist of both theoretical and practical sections:

**Theory**

We will discuss the basic theoretical principles of visualization. The following principles will be covered:
- Graph types (tabular data): univariate vs. bivariate; categorical vs. continuous
- Color palettes: perceptually uniform vs. non-uniform; discrete vs. continuous
- Graph types (neuroimaging data): statistical maps, connectomes, etc.

This part includes several interactive elements to lead students to form their own understanding of the material. The code is already provided, but we will not dwell on it.

**Practice**

We will put the theoretical principles into practice as we go. We will use the following libraries:
- matplotlib
- seaborn
- ptitprince
- plotly
- nilearn

Students will be required to modify the provided code to understand the role of various visualization parameters and to answer specific questions based on what we cover in class or from provided references (e.g., matplotlib documentation).
<!-- #endregion -->

<div class="alert alert-block alert-warning">
The <b>yellow boxes</b> contain questions/exercises for students to answer.
</div>


<div class="alert alert-block alert-info">
The <b>blue boxes</b> contain additional information about the datasets and functions used.
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

## Load the data

### Brain development fMRI dataset
For this tutorial, we will use the brain development fMRI dataset, which includes phenotypic data as well fMRI data collected from children and adults during movie watching ([Richardson et al., 2018](https://doi.org/10.1038/s41467-018-03399-2)).

**Note:** if you have already downloaded the data, you can change the path in the cell below to point to the appropriate directory.

```python
development_dataset = datasets.fetch_development_fmri(data_dir='data/')
```

```python
development_dataset['func']
```

### Datasaurus
In the first part of the tutorial, we will also briefly use the dataset datasaurus. If you want to be able to execute the cells using that dataset, you will have to download the data from [kaggle](https://www.kaggle.com/datasets/tombutton/datasaurusdozen).

**Note:** change the path in the cell below to point to the directory where you downloaded the data.

```python
# Credit: Alberto Cairo (original datasaurus), and Justin Matejka and George Fitzmaurice (datasaurus dozen)
data = pd.read_csv("data/datasaurus.csv") 
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## A picture is worth a thousand words
- Visualizations help us better understand the complexity of our data
- They allow us to support our findings
- And they help us share a message
<!-- #endregion -->

```python
# rcParams allows us to modify the global parameters of our figures
# Here, we are simply ensuring that values between (-8,000,000, 8,000,000) are not shown in scientific notation
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
### "Never trust summary statistics alone; always visualize your data"
<p style="margin-top: -1em;"><i>Alberto Cairo</i></p>

Python offers a wide range of libraries for visualizing our data:
- High-level vs. low-level
- Static images vs. interactive plots
- General-purpose libraries (e.g., Matplotlib, Seaborn, Bokeh, and Plotly)
- Domain-specific libraries (e.g., Nilearn)

But with great power comes great responsibility...
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
plt.title("What could you say about 'A' and 'B' if you only look at the figure?")
plt.show()
```

```python
plt.bar(data['Categories'], data['Values'])
plt.ylim([5500000, 8000000])
    
plt.title("What do you observe?")
plt.show()
```

```python
plt.bar(data['Categories'], data['Values'])

# Ajoute les valeurs au-dessus des bars
for i, v in enumerate(data['Values']):
    plt.text(i, v + 1, str(v), ha='center', va='bottom')
    
plt.title("Is that better?")
plt.show()
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## A graph for every data type!

There are several types of graphs. The chosen chart type depends on the variables you want to visualize:
- Univariate visualization:: **continuous variable** vs **categorical**
- Bivariate visualization: **categorical x categorical** vs **categorical x continuous** vs **continuous x continuous**
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Univariate visualizations

For a **continuous variable**, we can visualize its distribution:
- Histogram
- kde plot
- Strip plot
  
For a **categorical variable**, we can visualize the quantity of observations for each category:
- Bar plot
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
<div class="alert alert-block alert-danger">
If the `brain development fMRI dataset` is not under the <i>data/</i> folder, change the path in the cell below for the appropriate one.
</div>
<!-- #endregion -->

```python
# Retrieve participants data
participants = pd.read_csv('data/development_fmri/development_fmri/participants.tsv', sep='\t')

# Let's check what we have here
participants.head()
```

```python
# Let's check our data types
participants.dtypes
```

<div class="alert alert-block alert-info">
The dataset contains continuous variables (e.g., `Age`, `ToM Booklet-Matched`, `FB_Composite`) and categorical variables (e.g., `AgeGroup`, `Child_Adult`, `Gender`). `ToM Booklet-Matched` represents a score on a task designed to assess Theory of Mind—the ability to attribute mental states (beliefs, desires, emotions, intentions) to oneself or others.
</div>

```python
# Let's look at the descriptive statistics related to the `Age` variable
print(participants['Age'].describe())
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
#### Histogram

A histogram allows us to visualize the distribution of a given variable in a discrete manner by grouping its values into consecutive intervals (bins). This provides the frequency (i.e., the number of observations) within each of these intervals.

You can generate a histogram in Matplotlib using the [hist function](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html).
<!-- #endregion -->

```python jp-MarkdownHeadingCollapsed=true
# Let's visualize the distribution of `Age`
plt.hist(participants['Age'])
# Adding a title
plt.title('Age distribution')
# Adding a title for the x and y label
plt.xlabel('Age')
plt.ylabel('Frequency')
```

<div class="alert alert-block alert-info">
<b>The science behind the number of bins – Part 1</b>
<br>The <i>hist</i> function in <i>matplolib</i> groups data into 10 bins by default. However, an inappropriate number of bins (either too small or too large) will not represent the distribution accurately.
</div>


<div class="alert alert-block alert-warning">
<b>Bins in practice!</b>
<br>Modify the value of the `bins` parameter in the cell below to determine how many bins would most accurately reflects our `Age` variable.
</div>

```python
help(plt.hist)
```

```python
# Change the value of `bins` (ex. bins=10)
# Let's visualize the distribution of `Age`
plt.hist(participants['Age'], bins=10)
# Adding a title
plt.title('Age distribution')
# Adding a title for the x and y label
plt.xlabel('Age')
plt.ylabel('Frequency')
```

<div class="alert alert-block alert-info">
<b>The science behind the number of bins – Part 2</b>
<br> Several different rules exist for calculating the optimal number of bins, as well as the width of the intervals to be used for each. You can use these rules directly within the matplotlib <i>hist</i> function! Simply pass one of the valid rules as the bins parameter.
</div>

```python
help(plt.hist)
```

```python
# Let's visualize the distribution of `Age` (ex. bins='fd')
plt.hist(participants['Age'], bins='fd')
# Adding a title
plt.title('Age distribution')
# Adding a title for the x and y label
plt.xlabel('Age')
plt.ylabel('Frequency')
```

#### kde plot

The **Kernel Density Estimation (KDE)** allows us to visualize he distribution of our variable in a continuous (rather than discrete) manner by estimating a density function. However, much like bin size in a histogram, kernel density estimation is sensitive to the bandwidth!

```python
# To visualize the kernel density estimation, we will use the `kdeplot` function in `seaborn`
sns.kdeplot(participants['Age'])
```

<div class="alert alert-block alert-info">
<b>KDE plot y axis - Part 1</b>
<br> You’ve likely noticed that the y-axis values now range from 0 to 0.08 (as opposed to 0 to 50 for our histogram). In a KDE plot, the y-axis represents density, which is the probability per unit of the variable on the x-axis. In other words, it shows how "dense" the data is for a given x-value. Therefore, the peaks in this type of figure represent a higher density of points for a specific range of values (i.e., a higher probability of observing a value), while the troughs represent a lower density of points.
</div>


<div class="alert alert-block alert-info">
<b>Take note!</b>
<br>Since this type of graph provides a continuous estimation, it might suggest that certain data points exist when, in reality, they do not.
</div>

```python
# We can also overlap a histogram with a kde plot in `seaborn` using the `histplot` function
# (kde=True, bins='fd', edgecolor=None)
sns.histplot(
    participants['Age'],
    kde=True,
    bins='fd',
    edgecolor=None
)
```

<div class="alert alert-block alert-info">
<b>KDE plot y axis - Part  2</b>
<br> When overlaying a histogram with a KDE using `seaborn`, you will notice that the y-axis shows the frequency. However, the KDE curve remains a density curve. This occurs because `seaborn` scales the density curve to match the histogram by multiplying the curve by the number of observations and the bin width.
</div>


#### Univariate strip plot

**Histograms** and **KDE plots** allow us to visualize a variable's distribution in a discrete or continuous manner, respectively. However, these types of visualizations do not allow us to see the raw data itself.

The **strip plot** allows us to visualize each individual data point. This can make it easier to identify the presence of outliers within our data. However, a scatter plot is not suitable if we have too many data points.

```python
# Using seaborn `stripplot` function
sns.stripplot(
    x=participants['Age']
)
```

<div class="alert alert-block alert-warning">
<b>Strip plot reproducibility</b>
<br> Try to reproduce the strip plot for the `Age` variable in the two cells below. What do you notice?
</div>

```python
sns.stripplot(
    x=participants['Age']
)
```

```python
sns.stripplot(
    x=participants['Age']
)
```

<div class="alert alert-block alert-info">
<b>The `jitter` parameter</b>
<br>You might have noticed that the two scatter plots you generated from the same variable are not exactly identical. This happens because seaborn uses numpy.random to calculate the <i>jitter</i>. To make the <i>jitter</i> calculation reproducible, you can set a <i>seed</i> beforehand using `np.random.seed()`.
</div>

```python
np.random.seed(10)
sns.stripplot(
    x=participants['Age']
)
```

```python
np.random.seed(10)
sns.stripplot(
    x=participants['Age']
)
```

#### Bar plots

The **bar plot** allows you to visualize and compare categorical variables by showing the frequency of different values or simply the values themselves. This type of graph is useful if you want to, for example, compare the number of people per group.

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

### Summary

| Graph type | Data type | Summary |
| --- | --- | --- |
| Histogram | Continuous | To visualize our data distribution in a discrete way |
| KDE plot | Continuous | To visualize our data distribution in a continuous way |
| Strip plot | Continuous | To visualize each data point individually |
| Bar plot | Categorical | To compare frequencies between groups/categories |



### Bivariate Visualizations

For a **continuous variable** x **continuous variable**, we can use:
- Scatter plot
- Bivariate KDE
- Hexplot
- Joint plot
- Heat map

For a **categorical variable** x **continuous variable**, we can use:
- Box plot
- Violin plot
- Scatter plot
- Point plot
- Rain cloud plot
- Bar plot... (?)


#### Scatter plot - Continuous variable x continuous variable

```python
help(plt.scatter)
```

```python
# Let's look at the relation between 'Age' and 'ToM Booklet-Matched' with matplotlib `scatter` function
plt.scatter(
    x='Age',
    y='ToM Booklet-Matched',
    data=participants
)
plt.xlabel('Age')
plt.ylabel('ToM score')
plt.title('Relationship between Age and ToM score')
```

<div class="alert alert-block alert-warning">
<b>What do you notice?</b>
<br> Take the time to observe the figure that we have just generated.
</div>

```python
participants.groupby(['Child_Adult'])['ToM Booklet-Matched'].mean()
```

<div class="alert alert-block alert-info">
<b>Missing values</b>
<br>The `scatter` function in `matplotlib`, as well as the `regplot` function in `seaborn` automatically remove the missing values.
</div>


<div class="alert alert-block alert-info">
<b>The `regplot` function</b>
<br>The `regplot` function in `seaborn` allows you to visualize the scatter plot while fitting a linear regression model to the data!
</div>

```python
# Let's look at the relation between 'Age' and 'ToM Booklet-Matched' with seaborn `regplot` function
sns.regplot(
    x='Age',
    y='ToM Booklet-Matched',
    data=participants
)
```

<div class="alert alert-block alert-warning">
<b>Fitting the regression model</b>
<br>A linear regression does not seem to be the best way to model the relationship between our `Age` variable and our `ToM Booklet-Matched` variable. Change the value of the order parameter in the regplot function to 2 to check the fit of this model.
</div>

```python
sns.regplot(
    x='Age',
    y='ToM Booklet-Matched',
    data=participants,
    order=2
)
```

<div class="alert alert-block alert-info">
<b>What if we are adding another variable!</b>
<br>We can visualize interactions between multiple variables via the `hue` parameter in the `lmplot` function in `seaborn`. In the cell below, we will explore the relation between `Age` (<b>x axis</b>) and `ToM Booklet-Matched` (<b>y axis</b>) based on the gender (<b>hue</b>).
</div>

```python
# Let's look at the relation between 'Age' and 'ToM Booklet-Matched' with seaborn `lmplot` function
sns.lmplot(
    x='Age',
    y='ToM Booklet-Matched',
    data=participants,
    hue='Gender'
)
```

#### Bivariate KDE plot and hex plot - Continuous Variable x continuous Variable

When we have a large number of data points and want to visualize the distribution of points across two variables, we risk having significant overplotting. In this case, it can be difficult to properly visualize the distribution of our data with a scatter plot. It is therefore possible to use other types of graphs:

The **bivariate KDE plot** allows us to visualize how two variables are distributed in a two-dimensional space. Each contour represents a density zone. The closer the contours are, the higher the density—meaning where the data is most concentrated.

The **hex plot** allows us to visualize the data point density in a discrete manner. It is essentially the equivalent of a histogram, but for visualizing two variables instead of one. Darker areas represent zones of high density.

&#x26a0; Bivariate kernel density estimation is more computationally demanding compared to the hex plot.

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

#### Strip plot - Continuous variable x categorical variable

We have already discussed the **strip plot** in the context of univariate visualization, but we can also use it to look at mulitple groups/categories simultaneously.

```python
order = sorted(participants.AgeGroup.unique())[:-1]

sns.stripplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'], 
    order=order
)
```

<!-- #region -->
#### Box plot - Continuous variable continue x categorical variable


Box plots allow us to visualize the distributions of one or multiple groups of continuous variables (e.g., age distributions across different experimental groups). The different components of the box plot represent different descriptive statistics:
- The line in the box represens the median.
- The extremities of the box (inferior-Q1 and superior-Q3 quartiles) represent the range where 50% of the data is located.
- The whiskers (i.e., the lines outside of the box) capture the range of the rest of the data.
- The points represent outliers—values greater than 1.5 x interquartile range (i.e., Q3-Q1) + Q3 or lower than Q1 - 1.5 x interquartile range.
<!-- #endregion -->

```python
sns.boxplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'],
    order=order
)
```

#### Violon plot - Continuous variable x categorical variable

The **violin plots allow** us to visualize the data distribution by using the density curves (aka the **kde** curves). The width of each curve corresponds to the approximate frequency of the points for each region (values on the y-axis).

```python
sns.violinplot(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'], 
    order=order
)
```

#### Raincloud - Continuous variable x categorical variable

Why choosing between strip plot, box plot and violin plot when we can do them all at one! That's what **raincloud plots** allow us to do.

*Note :* this type of graph is not natively integrated into `seaborn` or `matplotlib`. We will need to use the `ptitprince library` that we imported at the beginning of the tutorial (`import ptitprince as pt`).

*Resource :* for more examples using raincloud plot, please refer to this [ptitprince tutorial](https://github.com/pog87/PtitPrince/blob/master/tutorial_python/raincloud_tutorial_python.ipynb).

```python
pt.RainCloud(
    x='AgeGroup', 
    y = 'ToM Booklet-Matched', 
    data = participants[participants.AgeGroup!='Adult'], 
    order=order,
    bw=0.6
)
```

#### Point plot - Continuous variable x categorical variable

The **point plot** allows us to compare means (or any other descriptive statistic) between groups while showing the uncertainty (e.g., 95% CI, standard deviation, etc.). This type of graph is useful to show trends between different categories or between different time points (e.g., if we have longitudinal measurements).

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

#### Bar plots for bivariate visualizations?

You may have already seen bar plots used to represent a continuous variable. However, this type of visualization is not recommended for this kind of variable:
- It only allows for the visualization of certain descriptive statistics (e.g., the mean) without providing any information about the distribution of our variable.
- If you absolutely must use a bar plot, overlay it with a scatter plot!

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

## And what if we added a little color to our figures?


### Perceptually Uniform vs. Non-Uniform Color Palettes
- Colors are perceived based on their hue (orange, red, green, etc.) and their luminosity (lightness vs. darkness of a hue).
- The characteristics of our photoreceptors mean that we do not process the light spectrum uniformly.
- The majority of the photoreceptors that allow us to see colors (cones) process long wavelengths (i.e., red, orange, yellow).
- Therefore, we do not perceive variations in green-blue hues as well as we do perceive variations in yellow-red hues.

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
<b>What do you notice?</b>
<br>Compare the colored image using the `hsv` and `batlow` color palettes with the black and white (grayscale) version.
</div>


<div class="alert alert-block alert-info">
<b>You don't have to throw away the rainbow</b>
<br> Google developed a perceptually uniform rainbow colormap called `turbo`, which is available on `matplotlib`. For more details about this colormap, please refer to this <a href="https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/">Google Research blog</a>.
</div>


### Discrete vs. Continuous Color Palettes

Color palettes can be either continuous (like those we saw above) or discrete. Discrete color palettes are used to visualize categorical variables—where categories have no inherent order (e.g., Children with COVID vs. children without COVID vs. adults with COVID vs. adults without COVID).


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
<b>Discretizing Continuous Palettes</b>
<br>In the cells below, we saw that it is possible to discretize a continuous palette by specifying the number of colors we want. However, as mentioned, discrete palettes are typically used to visualize categories that have no inherent order. Therefore, using an ordered discrete palette is not always necessary.
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

### Diverging color palettes

Diverging color palettes are useful when we have an interpretable central value. Diverging palettes can be applied to both discrete and continuous scales. For example:
- **Discrete diverging palettes**: If we have data collected using a Likert scale. Our values might range from 'Strongly Disagree' to 'Strongly Agree,' with 'Neutral' as the central value. In this case, we could visualize the values on the left (from 'Strongly Disagree' to 'Neutral') in shades of blue and the values on the right (from 'Neutral' to 'Strongly Agree') in shades of orange/red.
- **Continuous diverging palettes**: If we have negative and positive values (e.g., correlation coefficients), we could use zero as the central value. Negative values could be represented in shades of blue and positive values in shades of orange/red.


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

### Universally interpretable color palettes

We talked about the perceptual uniformity of color palettes, but it is also important to consider the use of colors that are perceived universally. In fact, using certain color combinations can make it difficult for people with color vision deficiency to distinguish between data points. A few tips:
- Avoid red-green combinations since the most common form of color blindness
- Vary lightness and saturation to make sure the data remains readable even in greyscale
- Use the colorblind-friendly palettes provided by `matplotlib`, `seaborn` and `cmcrameri`

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
<b>More than colors</b>
<br> Choosing the right color palette is important, but there are also other strategies we can use to make our figures more accessible and easily interpretable. Instead of using only different colors to distinguish between categories, we could use <b>different marker shapes</b> (e.g., circle vs triangle). If our figure includes lines to, for example, illustrate the trajectory of a variable over time, we could use <b>different line style</b> (e.g., solid vs dashed). For more examples, see <a href="https://www.datylon.com/blog/data-visualization-for-colorblind-readers">"The best charts for color blind viewers"</a>.
</div>


## Anatomy of a figure

We have already seen how to add or modify certain elements of our figures, such as the title and axis labels. In this section, we will discuss the art of figure-making in more detail. We will cover:
- Subplots
- Spines 
- Ticks
- Grid
- Legend


### Subplots

Up until now, we have primarily created our figures by directly calling certain functions from matplotlib and seaborn. However, if we want to create a figure with multiple panels (i.e., columns and/or rows), we will need to use subplots.


#### matplotlib

```python
# Let's look at what we've used before. This code generates two separated figures.
plt.hist(participants['Age'], bins='fd')
plt.title("Age Distribution")
plt.xlabel('Age')
plt.ylabel('Frequency')
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
# If we want to produce a single figure with these two graphs, we will have to use subplots
# Function syntax: plt.subplots(n_rows, n_cols, *)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(participants['Age'], bins='fd')
axes[0].set_title("Age Distribution")
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

axes[1].scatter(participants['Age'], participants['ToM Booklet-Matched'])
axes[1].set_title("Relation between Age and ToM scores")
axes[1].set_xlabel('Age')
axes[1].set_ylabel('ToM Booklet-Matched')
```

```python
# The subplots can even be used if we only have one graph
fig, axes = plt.subplots(figsize=(6, 4))
axes.hist(participants['Age'], bins='fd')
axes.set_title("Age Distribution")
axes.set_xlabel('Age')
axes.set_ylabel('Frequency')
```

#### seaborn

```python
# With searbon
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

### Spines
The spine (or border) refers to the lines around the plotting area.

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
ax.set_title("Age Distribution")
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')

ax.spines[['right', 'top']].set_visible(False) # ax.spines.top.set_visible(False)
```

### Ticks

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Age Distribution")
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')

ax.tick_params(
    axis='both', # Modification applied to both axes (x and y)
    which='major', # Modification on 'major', 'minor' ou 'both' ticks
    length=10, # Ticks length
    width=2, # Ticks width
    color='purple', # Ticks color
    labelsize=12, # Label size
    labelcolor='darkorange', # Label color
    labelrotation=45
)

#from matplotlib.ticker import MultipleLocator
#ax.xaxis.set_minor_locator(MultipleLocator(2))
```

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(participants['Age'], bins='fd')
ax.set_title("Age Distribution")
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')

plt.tick_params(
    axis='x',
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
```

<div class="alert alert-block alert-warning">
<b>Modify the code</b>
<br> Based on the code provided in the previous cells, modify the cell below to generate a figure with the following characteristics:
<li>
    No top or right spines
</li>
<li>
    No tick marks on the y-axis
</li>
<li>
    Minor ticks on the x-axis at intervals of 1
</li>
<li>
    X-axis labels rotated to 90 degrees
</li>
<li>
    Titles for all axes as well as for the figure
</li>

```python
# Add your code here
# ...
```

### Grid

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

ax.set_axisbelow(True) # Equivalent to the `zorder` parameter
```

### Legend

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

### Default parameters

The default parameters for all elements of our figures (e.g., font, text size, line width, etc.) are defined in the `rcParams` object. These values can be modified, allowing us to apply a consistent figure style from one figure to another.

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
axes.set_title("Age Distribution")
axes.set_xlabel('Age')
axes.set_ylabel('Frequency')
```

```python
# To reset the default values
plt.rcdefaults()
```

<div class="alert alert-block alert-info">
<b>Choose your style</b>
<br> It is also possible to use predefined styles using the `style.use` function in `matplotlib`. You can consult the list of available styles <a href=https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>here</a>. See also the <a href='https://matplotlib.org/stable/users/explain/customizing.html'>matplotlib documentation</a> to learn more about style sheets and `rcParams`.
</div>

```python
print(plt.style.available)
```

<div class="alert alert-block alert-info">
For example, 'ggplot' is a style that we can use to obtain figures similar to those generated by the `ggplot` library in R.
</div>

```python
plt.style.use('ggplot')
```

```python
fig, axes = plt.subplots(figsize=(6, 4))
axes.hist(participants['Age'], bins='fd')
axes.set_title("Age Distribution")
axes.set_xlabel('Age')
axes.set_ylabel('Frequency')
```

## Interactive graphs

Static figures are good, but interactive figures are better! In fact, interactive graphics offer us the opportunity to explore our data in ways that wouldn't be possible with static figures and to create dashboards (see [documentation](https://dash.plotly.com/?_gl=1*j1jto0*_gcl_au*MTY3MTkxNTc4MS4xNzczOTMwNTAw*_ga*MTM4NzMyMTAxMy4xNzczOTMwNTAy*_ga_6G7EE0JNSC*czE3NzM5MzA1MDIkbzEkZzAkdDE3NzM5MzA1MTAkajUyJGwwJGgw)). We can add information to our figures without cluttering them.

The two main Python libraries that allow us to create interactive figures are `plotly` and `bokeh`. The `plotly` library offers a high-level interface (`plotly.express`) that allows us to create figures in a single line of code, while `bokeh` offers more customization options. In this tutorial, we will only discuss `plotly.express`, but if you want to try `bokeh`, you can consult [the documentation](https://docs.bokeh.org/en/latest/)

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

## Visualizing brain images!

The `nilearn` library supports multiple functions to plot brain images (see [the list of functions](https://nilearn.github.io/stable/plotting/index.html)). In this section, we will see some of those function.

---

Based on the [MAIN tutorial](https://main-educational.github.io/intro_nilearn/machine-learning-with-nilearn.html)

```python
# Let's get our data
data = development_dataset.func
confounds = development_dataset.confounds
pheno = pd.DataFrame(development_dataset.phenotypic)
```

```python
data
```

```python
data[0]
```

```python
# Let's try visualizing our first file
plotting.view_img(data[0])
```

```python
img = nib.load(data[0])
img.shape
```

<div class="alert alert-block alert-info">
<b>Oups!</b>
<br> If we try to visualize our BOLD images, we get an error! This is perfectly normal, as our file contains 4D data—meaning that for every voxel (3D), we have a value for several points in time (repetition time; +1D). If we absolutely want to visualize these files, two options are available to us:
<ul>
    1. We can look at the activity across the whole-brain for a given time point
</ul>
<ul>
    2. We can look at the timeserie for a given voxel/parcel
</ul>   
</div>

```python
help(plotting.view_img)
```

```python
# Let's retrieve our first volume for our first participant
first_volume = image.index_img(data[0], 0)

plotting.view_img(first_volume, black_bg=False, cmap='turbo', symmetric_cmap=False)
```

```python
help(plotting.plot_stat_map)
```

```python
plotting.plot_stat_map(
    first_volume, 
    draw_cross=False,
    cut_coords=(0, 4, 22),
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
# Extract the timeseries for our first participant
time_series = masker.fit_transform(data[0], confounds=confounds[0])
```

```python
time_series.shape
```

```python
parcel = 0
plt.figure(figsize=(12,4))
plt.plot(time_series.T[parcel])
plt.title(f'Timeserie for parcel {parcel}')
plt.xlabel('Volumes')
plt.ylabel('Amplitude')
```

<div class="alert alert-block alert-warning">
<b>Visually compare the timeseries</b>
<br>Based on the code provided in the previous cell, modify the cell below to be able to compare the timeserie of <b>parcel 0</b> and <b>parcel 1</b>.
</div>

```python
help(plt.legend)
```

```python
# To complete
```

## Supplementary resources

- [`seaborn` tutorial on visualizing distributions of data](https://seaborn.pydata.org/tutorial/distributions.html)
- [Python Graph Gallery](https://python-graph-gallery.com/)
- [Mastering data charts: A comprehensive guide to visualization](https://www.atlassian.com/data/charts)
- [Common caveats to avoid](https://www.data-to-viz.com/caveats.html)
- [Nilearn visualization functions - (f)MRI data](https://nilearn.github.io/dev/plotting/index.html)
- [MNE python visualization tutorials - EEG/MEG data](https://mne.tools/stable/auto_tutorials/visualization/index.html)
- [DIPY visualization tutorials - Diffusion MRI data](https://docs.dipy.org/stable/examples_built/index#visualization)
