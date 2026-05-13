# Python for data visualization and model interpretation

Training module on data visualization using Python tools

## Downloading the data

For this tutorial, we will use two different datasets.

### Datasaurus (optional)

We will use this dataset solely to illustrate certain theoretical concepts. Downloading this dataset is therefore optional.

To download the data:

https://www.kaggle.com/code/tombutton/datasaurus-dozen/input

### Brain development fMRI

We will download this dataset using `nilearn`. Follow the instructions in the notebook.

## Installation

1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Check `uv` installation

```bash
uv -h
```

3. Create a virtual environment

```bash
uv venv visu-env --python=3.10
```

4. Activate your environment

```bash
source visu-env/bin/activate
```

5. Install the dependencies

```bash
uv pip install -r requirements.txt
```

## Configure the virtual environment as a Jupyter notebook kernel

In your terminal:

```bash
python -m ipykernel install --user --name=visu-env --display-name "Python cours visu"
```