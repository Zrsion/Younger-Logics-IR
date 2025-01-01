This project is based on the "nasbench_keras" repository by lienching 
(https://github.com/lienching/nasbench_keras), with modifications for 
converting nasbench models to Younger instances. 

NASBench 101 official repository: https://github.com/google-research/nasbench

To run this project, you need to have the following dependencies installed:

- **Python** version >= 3.11
- **TensorFlow** version >= 2.0

First, install Younger. 
```bash
git clone https://github.com/Jason-Young-AI/Younger
cd path/to/Younger
git checkout dev
pip install -e .
```

Next, install `nasbench_keras`. Navigate to the nasbench_keras directory:
```bash
cd path/to/nasbench_keras
pip install -e .
```


After installing the required packages, you can run the conversion code `convert2dag.py`.

The `models-path` must be set to `path/to/generated_graphs.json`. The `generated_graphs.json` can be generated using the official NASBench101 code.


