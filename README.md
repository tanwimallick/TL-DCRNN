# TL-DCRNN: Transfer Learning with Graph Neural Networks for Short-Term Highway Traffic Forecasting
TL-DCRN, a new transfer learning approach for DCRNN, where a single model trained on a highway network can be used to forecast traffic on unseen highway networks. Given a traffic network with a large amount of traffic data, our approach consists of partitioning the traffic network into a number of subgraphs and using a new training scheme that utilizes subgraphs to marginalize the location-specific information, thus learning the traffic as a function of network connectivity and temporal patterns alone. The resulting trained model can be used to forecast traffic on unseen networks. We demonstrate that TL-DCRN can learn from  San Francisco regional traffic data and can forecast traffic on the Los Angeles region and vice versa.


## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- tensorflow>=1.13.1
- pyaml


## Data Preparation
Download the traffic data files for entire California ['speed.h5'](https://anl.box.com/s/7hfhtie02iufy75ac1d8g8530majwci0), adjacency matrix  ['adj_mat.pkl'](https://anl.box.com/s/4143x1repqa1u26aiz7o2rvw3vpcu0wp) and distance between sensors ['distances.csv'](https://anl.box.com/s/cfnc6wryh4yrp58qfc5z7tyxbbpj4gek), and keep in the `data/input_files/` folder.

```bash
# Generate TFrecord dataset for 64 graph partitions

python hdf_to_tfrecord.py --config_filename=input_files/tf_record_config.yaml
```
The script will generate a ```data/TFrecords/``` folder with the train, test, and validation dataset for 64 partitions

## Model Training

```bash
# Run the TL-DCRNN model

python dcrnn_train.py --config_filename=data/dcrnn_config_32transfer.yaml
```
The generated prediction of TL-DCRNN will be in ```data/results/```

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{mallick2021transfer,
  title={Transfer learning with graph neural networks for short-term highway traffic forecasting},
  author={Mallick, Tanwi and Balaprakash, Prasanna and Rask, Eric and Macfarlane, Jane},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={10367--10374},
  year={2021},
  organization={IEEE}
}
```
