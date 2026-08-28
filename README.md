# Voxelium-apex 

This repository contains the voxelium-apex codes -- a package for cryoEM/cryoET heterogeneity reconstruction analysis and visualization.

## Installation 
### Install from PyPI
First create and activate a Conda or standard Python virtual environment. Then install the prebuilt wheels distributed via PyPI:
```pip install voxelium-apex```


### Build from source
After cloning the repository and navigating (`cd`) into the project directory, first create and activate a Conda or standard Python virtual environment.
```pip install .```


## 3D Spectral Heterogeneity Analysis (SHA)
Run `vxm-apex -h` to see a list of modules.
To run the analysis, the sha3D module can be run as follows:

```vxm-apex sha3d <input_star_data> <log_directory> --gpu 0```

Here, `<input_star_data>` is an input STAR-file containing all the particles with CTF and pose parameters set.
`<log_directory>` will contain the results of the job. 

NOTE: Adding `--preload` speeds things up considerably, assuming the dataset fits in memory.

NOTE: You need to install extension for this, see above.

## SHA3D Visualization

To visualize the results run:

```vxm-apex sha3d_viewer <log_directory>```

In the above, `<log_directory>` is the path to the directory containing the results of the SHA3D analysis, see above.


## 🤝 Contributor covenant code of conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@biohub.org.

Responsible Use: We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with the model.

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@biohub.org.