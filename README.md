# MTY4 - Emergent Phenomena
This is the github repository for a minicourse created by Sergio Eraso and Jackie Hernandez for the 2025 edition of Clubes de Ciencia Mexico 2025 in Monterrey. The course is an introduction to scientific computing and statistical physics.

# Self-Study Guide

Although this repository was originally built for a taught minicourse, everything you need to work through the material on your own is here. This section is for newcomers who want to use the repo for self-study.

## What you'll learn

The course is a hands-on introduction to **scientific computing** in Python and to **statistical physics**, with a special focus on *emergent phenomena* — the idea that simple rules followed by many interacting parts can give rise to surprisingly complex collective behavior.

The material is split into two kinds of Jupyter notebooks:

**Foundations** — the tools you'll use throughout:
- `NB00` Python Basics Review — a quick refresher on Python syntax, functions, loops, and control flow.
- `NB01` Introduction to Scientific Computing — working with `numpy` arrays, broadcasting, indexing, and slicing.
- `NB02` Data Visualization — making plots and figures with `matplotlib`.
- `NB03` Probability — the probability concepts that underpin the physics.

**Activities** — applying those tools to real models from statistical physics:
- `A0` Random Walks — stochastic processes and Brownian motion.
- `A1` Cellular Automata — simple local rules that produce complex global patterns.
- `A2` Ising Model — a classic model of phase transitions and magnetism.
- `A3` Hopfield Network — a neural network for associative memory, built on the same physics as the Ising model.

> **Note:** This course assumes you already know the basics of programming. It is not designed to teach programming from scratch. If you're brand new to coding, `NB00` and the resources in [`suggested_readings.md`](suggested_readings.md) are a good place to begin.

## How the notebooks are organized

There are two folders of notebooks:

- **`Empty Notebooks/`** — the starting point. These are the notebooks to work through yourself, with the code left for you to fill in (the activity notebooks are the `A0`–`A3` files, and `NB00`–`NB02` have "Live" versions to code along with).
- **`Complete Notebooks/`** — the same notebooks with full sample solutions and explanations (these end in `C`, e.g. `NB00C`, `A0C`). Use these to check your work or if you get stuck.

The recommended workflow is to attempt each notebook in `Empty Notebooks/` first, then compare against the matching solution in `Complete Notebooks/`.

## Getting started

You have two easy ways to run the notebooks. **If you just want to start as quickly as possible, use Google Colab — you don't have to install anything.**

### Option 1 — Google Colab (no installation)

Colab runs the notebooks on Google's servers with all the required packages already installed.

1. Go to [Google Colab](https://colab.research.google.com/).
2. Choose **File → Open notebook → GitHub**, paste the URL of this repository, and pick the notebook you want.
3. Alternatively, download a notebook from this repo (see Option 2) and upload it with **File → Upload notebook**.

> **Important — making `simulation_functions` available in Colab.** The activity notebooks (`A1`, `A2`, `A3`) import helper code from the [`simulation_functions/`](simulation_functions) folder. When you open a notebook in Colab, that folder isn't there automatically, so the imports will fail until you add it. Colab's working directory is `/content`, so the folder needs to live at **`/content/simulation_functions`**. The easiest way to get it there is to run this in a cell at the top of the notebook:
>
> ```python
> # Download the repo and copy the helper package into Colab's working directory (/content)
> !git clone https://github.com/SergioEJr/CdCMX25.git
> !cp -r CdCMX25/simulation_functions /content/
> ```
>
> After running that once, `import simulation_functions.ising_functions` (and the other helper imports in the notebooks) will work. You can confirm the folder is in place by opening the **Files** panel on the left — you should see `simulation_functions` sitting directly under `/content`.

### Option 2 — Run locally with Anaconda

If you'd rather run everything on your own machine:

1. **Get the files.** The simplest option is to use the green **Code → Download ZIP** button on the GitHub page and unzip it. If you're comfortable with git, you can instead `git clone` the repo (see "Cloning vs. forking" below).
2. **Install Python.** We recommend the [Anaconda distribution](https://www.anaconda.com/download), which comes with Jupyter and most scientific packages.
3. **Install the packages used in the course** (most come with Anaconda already):
   ```bash
   pip install numpy matplotlib pillow tqdm
   ```
4. **Launch Jupyter** (e.g. `jupyter notebook` or `jupyter lab` from a terminal), navigate to the repo folder, and open a notebook from `Empty Notebooks/`.

The activity notebooks import helper code from the [`simulation_functions/`](simulation_functions) folder, so keep that folder alongside the notebooks if you move files around.

### Cloning vs. forking

For self-study, **the easiest path is to just download the ZIP (or clone) and start editing** — you don't need a GitHub account for that.

If you do have a GitHub account and want to keep your own copy of your work (including your solutions) under version control, **fork** the repository instead. A fork gives you your own copy of the repo on GitHub that you can commit and push your changes to, while still being able to pull in any future updates from the original. Cloning, by contrast, just copies the files to your computer without giving you a place to save your changes online.

## A suggested path

1. Skim `NB00` to refresh your Python.
2. Work through `NB01` and `NB02` to get comfortable with `numpy` and `matplotlib`.
3. Do `NB03` for the probability background.
4. Tackle the activities in order: `A0` → `A1` → `A2` → `A3`.
5. Check your work against the `Complete Notebooks/`, and dive into [`suggested_readings.md`](suggested_readings.md) whenever you want to go deeper on a topic.

Have fun, and don't hesitate to reach out with questions!

# Contact Information
For mistakes or questions please contact:\
Sergio Eraso: sergioerasojr@gmail.com\
Jackie Hernandez: lijaheay@gmail.com
