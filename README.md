<img src="figs/method.svg" alt="method image" align="left" width="200">

## Inter-source interferometry with cross-correlation of coda waves

This repository contains the source code for the reproduction of results of the following publication:

Tom Eulenfeld (2020), Toward source region tomography with inter-source interferometry: Shear wave velocity from 2018 West Bohemia swarm earthquakes, *Journal of Geophysical
Research: Solid Earth, 125*, e2020JB019931, doi:[10.1029/2020JB019931](https://doi.org/10.1029/2020JB019931). [[pdf](https://arxiv.org/pdf/2003.11938)]

#### Preparation

1. Download or clone this repository.
2. The data is hosted on zenodo with doi:[10.5281/zenodo.3741465](https://www.doi.org/10.5281/zenodo.3741465). Copy the data into a new folder `data`.
3. Create empty folder `tmp`.
4. Install the the relevant python packages: `python>=3.7 obspy>=1.2 matplotlib=3.1 numpy scipy=1.3 statsmodels tqdm shapely cartopy pandas utm obspyh5`.

#### Usage

The scripts `xcorr.py`, `traveltime.py`, `vs.py` should be run in this order. The scripts `plot_maps.py`, `vpvs.py` and `load_data.py` can be run independently. The scope of each script should be clear after reading the publication and an optional docstring at the top.

Three files are located in `data2` folder. `2018_events_qc_mag1.9.txt` QC file was created by the author upon visual inspection of the envelope plots created with the `load_data.py` script. `2018_events_tw_mag1.9.json` time window file can be recreated from the QC file with the `load_data.py` script. Running `load_data.py` is optional. `ccs_stack_2018_mag>1.8_q3_10Hz-40Hz_Scoda_envelope_dist<1km_pw2.h5` cross-correlation file contains the intermediate result of inter-event cross-correlation functions. This file can be directly read with obspy via obspyh5 plugin. The cross-correlation file is also created by the `xcorr.py` script inside the `tmp` directory.

The `figs` folder contains all figures from the publication. Most will be recreated when running the scripts.