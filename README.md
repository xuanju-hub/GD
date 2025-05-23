
* a basic implementation of SmartGD model,
* a generator checkpoint trained for Stress majorization,
* and a demo notebook for model traning and evaluation.

## 🔗 Prior Work - DeepGD
SmartGD is inspired by our prior work, DeepGD, which is a novel graph drawing method aims to optimize arbitrary differentiable aesthetic metrics. Please check this [repository](https://github.com/yolandalalala/DeepGD) for more details.

Paper: https://ieeexplore.ieee.org/document/9476996

## Environment
This code has been tested on `python3.11` + `cuda12.1` + `pytorch2.4` + `pyg2.4`. 

## Configuration
The default hyper-parameters of the model have been configured to reproduce the best performance reported in the [SmartGD paper](https://ieeexplore.ieee.org/document/10224347). 

## Training & Evaluation
* This repo provides a demo notebook `smartgd_demo.ipynb` for model training and evaluation. With Nvidia A100, each training epoch takes 5 minutes on average. It takes up to 1000 epochs to completly converge.

* This repo includes a two model checkpoints
  * `generator_stress_only.pt`: This reproduces the result for Stress Minimization-only objective function reported in the paper.
  * `generator_xing_only.pt`: This reproduces the result for Crossings Minimization-only objective function reported in the paper.

* For evaluation on custom data, the easiest way is to subclass `RomeDataset` and override `raw_file_names` and `process_raw` methods.
    > **Caveat**: Even though the behavior of `process` do not need to be overriden, it is required to have a dummy `def process(self): super().process()` defined in the subclasses to make it work properly. For details, please refer to `pyg.data.InMemoryDataset` [documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset).

* For easy benckmarking with SmartGD and a comprehensive set of other baseline methods, please check [GraphDrawingBenchmark](https://github.com/yolandalalala/GraphDrawingBenchmark). This benchmark repo automatically evaluates all common graph drawing metrics on the same train/test split for Rome dataset as used in the SmartGD paper.

## Citation
If you used our code or find our work useful in your research, please consider citing:

