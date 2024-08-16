# FSCVP: Feature Sharing Connection for Efficient Spatiotemporal Prediction

This is a pytorch implementation of FSCVP: Feature Sharing Connection for Efficient Spatiotemporal Prediction

## Installation

```shell
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

## Train

```shell
bash tools/prepare_data/download_taxibj.sh
python tools/train.py -d taxibj -c configs/taxibj/FSCVP.py --opt adamw --epoch 50 --ex_name taxibj_fscvp 
```

## Structures

- `data/` Where downloaded data will be stored.
- `configs/taxibj/FSCVP.py` FSCVP parameters for training.
- `openstl/methods/FSCVP.py` Training method of FSCVP is contained.
- `openstl/models/fscvp_model.py` FSCVP model is contained.
- `openstl/modules/fscvp_module.py` FSCVP modules are contained.


## Acknowledgement
Thanks to [OpenSTL](https://github.com/chengtan9907/OpenSTL) for providing great code

