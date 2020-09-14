# RA_FA_Cardiac

**Disentangled Representations for Domain-generalized Cardiac Segmentation** [[Paper]](https://arxiv.org/abs/2008.11514). In [M&Ms Challenge](https://www.ub.edu/mnms/) of [STACOM 2020](http://stacom2020.cardiacatlas.org/).

The repository is created by [Xiao Liu](https://github.com/xxxliu95), [Spyridon Thermos](https://github.com/spthermo), [Agisilaos Chartsias](https://github.com/agis85), [Alison O'Neil](https://www.eng.ed.ac.uk/about/people/dr-alison-oneil), and [Sotirios A. Tsaftaris](https://www.eng.ed.ac.uk/about/people/dr-sotirios-tsaftaris), as a result of the collaboration between [The University of Edinburgh](https://www.eng.ed.ac.uk/) and [Canon Medical Systems Europe](https://eu.medical.canon/). 

This repository contains the official PyTorch implementation of the Resolution Augmentation (RA) and Factor-based Augmentation (FA) methods proposed in the paper.

# System Requirements
* Pytorch 1.5.1 or higher with GPU support
* Python 3.7.2 or higher
* SciPy 1.5.2 or higher
* CUDA toolkit 10 or newer

# SDNet
In this repository, we train a SDNet [[code](https://github.com/agis85/anatomy_modality_decomposition), [paper]](https://arxiv.org/pdf/1903.09467.pdf) with our proposed Resolution Augmentation and Factor-based Augmentation in a semi-supervised manner.

# Resolution Augmentation
We propose to use random resampling to augment the original dataset such that the resolutions of all the data are equally distributed in a certain range.

# Factor-based Augmentation
We first pre-train a SDNet model to extract the anatomy and modality factors. Then mix the anatomy and modality factors to generate new images.

# Citation
If you find our metrics useful please cite the following paper:
```
@article{liu2020disentangled,
  title={Disentangled Representations for Domain-generalized Cardiac Segmentation},
  author={Liu, Xiao and Thermos, Spyridon and Chartsias, Agisilaos and O'Neil, Alison and Tsaftaris, Sotirios A},
  journal={arXiv preprint arXiv:2008.11514},
  year={2020}
}
```

# License
All scripts are released under the MIT License.
