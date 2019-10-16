# pytorch_divcolor

PyTorch implementation of Diverse Colorization -- Deshpande et al. "[Learning Diverse Image Colorization](https://arxiv.org/abs/1612.01958)"                 

This code is tested for python-3.5.2 and torch-0.3.0. Install packages in requirements.txt

The Tensorflow implementation used in the paper is [divcolor](https://github.com/aditya12agd5/divcolor)


Fetch data by

```
bash get_data.sh
```

Execute main.py to first train vae+mdn and then, generate results for LFW

```
python main.py lfw
```


current output dir is 'data/output/lfw/'

