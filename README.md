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

```bash
vxm-apex sha3d <log_directory> --input <input_star_data> --gpu 0
```

Here, `<input_star_data>` is an input STAR-file containing all the particles with CTF and pose parameters set.
`<log_directory>` will contain the results of the job. 

NOTE: Adding `--preload` speeds things up considerably, assuming the dataset fits in memory.

NOTE: You need to install extension for this, see above.

### Waving-spike example

The [Zenodo waving-spike dataset](https://zenodo.org/records/7182156) contains
100,000 noisy, CTF-applied projections spanning ten SARS-CoV-2 spike states
(SNR 0.1). Download and run it as follows:

```bash
curl -L -o waving_spike_SNR0.1.tar.gz \
  'https://zenodo.org/records/7182156/files/waving_spike_SNR0.1.tar.gz?download=1'
tar -xzf waving_spike_SNR0.1.tar.gz
cd waving_spike_SNR1-10

vxm-apex sha3d apex_waving_spike \
  --input refine3d/run_data.star --gpu 0 --preload --epochs 10
```

Run the command from the extracted dataset directory so that the relative image
paths in the STAR file resolve correctly. Expect the two-dimensional latent
space to organize the spike's closed-to-open motion, with the corresponding 3D
variation concentrated around the moving spike regions. View the result with
`vxm-apex sha3d_viewer apex_waving_spike`.

<!-- Add latent-space and reconstructed-structure figures here. -->

## SHA3D Visualization

To visualize the results run:

```vxm-apex sha3d_viewer <log_directory>```

In the above, `<log_directory>` is the path to the directory containing the results of the SHA3D analysis, see above.


## 🤝 Contributor covenant code of conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@biohub.org.

Responsible Use: We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with the model.

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@biohub.org.
