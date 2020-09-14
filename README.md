# RA_FA_Cardiac

![overview](./assets/images/overview.png)

**Disentangled Representations for Domain-generalized Cardiac Segmentation** [[Paper]](https://arxiv.org/abs/2008.11514).

This repository contains the official PyTorch implementation of the Resolution Augmentation (RA) and Factor-based Augmentation (FA) methods proposed in Disentangled Representations for Domain-generalized Cardiac Segmentation [https://arxiv.org/abs/2008.11514].

The repository is created by [Xiao Liu](https://github.com/xxxliu95)__\*__, [Spyridon Thermos](https://github.com/spthermo)__\*__, [Agisilaos Chartsias](https://github.com/agis85), [Alison O'Neil](https://www.eng.ed.ac.uk/about/people/dr-alison-oneil), and [Sotirios A. Tsaftaris](https://www.eng.ed.ac.uk/about/people/dr-sotirios-tsaftaris), as a result of the collaboration between [The University of Edinburgh](https://www.eng.ed.ac.uk/) and [Canon Medical Systems Europe](https://eu.medical.canon/).

# System Requirements
* Pytorch 1.5.1 or higher with GPU support
* Python 3.7.2 or higher
* SciPy 1.5.2 or higher
* CUDA toolkit 10 or newer

# Resolution Augmentation

run the following command:

```python compute_DC.py --root <path/to/extracted/tensors/and/vectors> --save </name/of/result/file>```

# Factor-based Augmentation

run the following command:

```python compute_IOB.py --root <path/to/extracted/tensors/and/vectors> --gpu <gpu_id> --save </name/of/result/file> ```

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
