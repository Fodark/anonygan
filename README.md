[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/XingGAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.7.10-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.8.1-red.svg)

## Contents
  - [AnonyGAN](#AnoynGAN)
  - [Installation](#Installation)
  - [Dataset Preparation](#Dataset-Preparation)
  - [Generating Images Using Pretrained Model](#Generating-Images-Using-Pretrained-Model)
  - [Train and Test New Models](#Train-and-Test-New-Models)
  - [Evaluation](#Evaluation)
  - [Acknowledgments](#Acknowledgments)
  - [Citation](#Citation)
  - [Contributions](#Contributions)

## AnonyGAN
**| [Paper]() |** <br> 
[**Graph-based Generative Face Anonymisation with Pose Preservation**]() <br> 
[Nicola Dall'Asen]()<sup>12</sup>, [Yiming Wang]()<sup>3</sup>, [Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>4</sup>, [Luca Zanella]()<sup>3</sup>, [Elisa Ricci](http://elisaricci.eu)<sup>23</sup>. 
<br><sup>1</sup>University of Pisa, Italy, <sup>2</sup>University of Trento, Italy, <sup>3</sup>Fondazione Bruno Kessler, Italy, <sup>4</sup>ETH Zürich, Switzerland.<br>
In [ICIAP 2021](https://www.iciap2021.org). <br>
The repository offers the official implementation of our paper in PyTorch.

## Installation

Clone this repo.
```bash
git clone git@github.com:Fodark/anonygan.git
cd anonygan/
```

Needed libraries are provided in the `requirements.txt` file. 

`pip install -r requirements.txt` should suffice.

## Dataset Preparation

- Download aligned CelebA [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Extract aligned version
- Compute landmarks and mask with the code provided in `preparation` (modify paths accordingly)

## Generating Images Using Pretrained Model

- Download pretrained model [here](https://drive.google.com/file/d/1FWMaBIQfm1-1fLy0ZG7eu--VD91OipJP/view?usp=sharing)
- Place it in `ckpts/anonygan.ckpt`
- Preprocess your images with the files in `preparation`
- Prepare a `.csv` files with columns `[from, to]` with condition and source images names
- Run `test.sh` modifying paths accordingly

## Train and Test New Models

- Same as using the pretrained model, for training modify `train.sh` accordingly

## Evaluation

`evaluation/automatic_evaluation` is the entry point, modify paths accordingly

## Acknowledgments

Graph reasoning inspired by [BiGraphGAN](https://github.com/Ha0Tang/BiGraphGAN)

## Citation

If you use this code for your research, please consider giving a star and citing our paper!

```
@inproceedings{dallasen2021anonygan,
  title={Graph-based Generative Face Anonymisation with Pose Preservation},
  author={Dall'Asen, Nicola and Wang, Yiming and Tang, Hao and Zanella, Luca and Ricci, Elisa},
  booktitle={International Conference on Image analysis and Processing},
  year={2021}
}
```

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Nicola Dall'Asen ([nicola.dallasen@phd.unipi.it](nicola.dallasen@phd.unipi.it)). 
