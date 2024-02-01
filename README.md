# Positron Alpha Testing

This repository currently contains the alpha version of the Positron library.

## Installation
After you've cloned the repo and `cd` into the project directory you first need to set up the proper Conda environment.
Use the `environment.yml` file to create a new environment called 'positron' with the right module installed, by running:

```conda env create -f environment.yml```

### Visualization only installation
If you only need to visualize reconstruction results (e.g. on you local computer) you can skip the building of the torch extensions. 
These are only needed on the computational nodes. First activate the new Conda environment:

```conda activate positron```

You can now install the positron library from inside the project directory by running:

```POSITRON_SKIP_EXT=TRUE pip3 install .```

Note that `POSITRON_SKIP_EXT` will skip installation of the torch extensions.

### Full installation

If you need to run reconstruction (e.g. on a computational node), you need to build and install the torch extensions.
You will need to have a CUDA toolkit installed for this that matches the pytorch version installed. 
Once you have that ready you can just run:  

```pip3 install .```