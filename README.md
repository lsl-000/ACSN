# 环境安装

This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [HINet](https://github.com/megvii-model/HINet) 

```python
python >=3.9
pytorch>=1.11.0
cuda 11.3
```

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

# 快速上手

* Datasets
  - LoLv2数据集
  - 预处理

```
# 先设置数据集路径，然后运行任何一条函数，生产lmdb文件
python basicsr/utils/create_lmdb.py
```

* Train:

```
# 设置yml文件的数据集路径和超参数
python basicsr/tarin.py -opt options/train/DarkIR.yml
```s
