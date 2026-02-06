# Positron Alpha Testing

This repository currently contains the alpha version of the Positron library.

### Full Installation
If you need to run reconstruction (e.g. on a computational node), you need to build and install the torch extensions.
You will need to have a CUDA toolkit installed for this that matches the pytorch version installed. 
Once you have that ready you can just run:  

```pip3 install .```


### Visualization-Only Installation
If you only need to visualize reconstruction results (e.g. on you local computer) you can skip the building of the torch extensions. 
These are only needed on the computational nodes. First activate the new Conda environment:

You can now install the voxelium-apex library from inside the project directory by running:

```POSITRON_SKIP_EXT=TRUE pip3 install .```

In the above, `POSITRON_SKIP_EXT` will skip installation of the torch extensions.


## 3D Spectral Heterogeneity Analysis (SHA)
Run `vxm-apex -h` to see a list of modules.
To run the analysis, the sha3D module can be run as follows:

```vxm-apex SHA3D <input_star_data> <log_directory> --gpu 0```

Here, `<input_star_data>` is an input STAR-file containing all the particles with CTF and pose parameters set.
`<log_directory>` will contain the results of the job. 

NOTE: Adding `--preload` speeds things up considerably, assuming the dataset fits in memory.

NOTE: You need to install extension for this, see above.

## SHA3D Visualization

To visualize the results run:

```vxm-apex sha3D_viwer <log_directory>```

In the above, `<log_directory>` is the path to the directory containing the results of the SHA3D analysis, see above.


## 🤝 Contributor covenant code of conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

Responsible Use: We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with the model.

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.