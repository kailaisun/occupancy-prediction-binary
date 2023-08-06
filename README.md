# OPTnet


Implementation of paper - [Multi-Sensor-Based Occupancy Prediction in a Multi-Zone Office Building with Transformer](https://www.mdpi.com/2075-5309/13/8/2002).


## Environment
- The code is tested on Windows 10 and Ubuntu 20.04.2, python 3, pytorch, sklearn,pandas.

## Train and test
```Bash
multi_source_LSTM.py

multi_source_DT.py

multi_source_transformer-copy.py
```

## Result

![1691305070202](https://github.com/kailaisun/occupancy-prediction-binary/assets/40592892/8165199a-69cd-42f7-a11d-3b77c9216b7c)

![1691305132225](https://github.com/kailaisun/occupancy-prediction-binary/assets/40592892/a3487800-bb73-4985-ade2-123e29e8bd0e)


## Citation
```Bash
@article{buildings13082002,
  author = {Qaisar, Irfan and Sun, Kailai and Zhao, Qianchuan and Xing, Tian and Yan, Hu},
  title = {Multi-Sensor-Based Occupancy Prediction in a Multi-Zone Office Building with Transformer},
  journal = {Buildings},
  volume = {13},
  year = {2023},
  number = {8},
  url = {https://www.mdpi.com/2075-5309/13/8/2002},
  issn = {2075-5309},
  doi = {10.3390/buildings13082002}
}
```

## Acknowledgement
This work is supported by the National Natural Science Foundation of China under Grant No. 62192751 and 61425027, in part, by the Key R&D Project of China under Grant No.
2017YFC0704100, 2016YFB0901900, by the 111 International Collaboration Program of China under Grant No. BP2018006, the 2019 Major Science and Technology Program for the Strategic Emerging Industries of Fuzhou under Grant No. 2019-Z-1, and, in part, by the BNRist Program under Grant No. BNR2019TD01009, and the National Innovation Center of High Speed Train R&D project (CX/KJ-2020-0006).


