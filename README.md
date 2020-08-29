# pepper
## Introduction
Protein Energy-based Pathogenicity PrEdictoR (PEPPER) is a machine learning method based on the protein structure energy to predict the pathogenicity of single amino acid variants (SAVs).

## Requirements
All needed packages are recorded in requirements.txt except PyRosetta [1].

### PyRosetta
PyRosetta is a Python-based interface to the Rosetta. It provide an convenient way to implement Rosetta molecular modeling suite.

To install Rosetta in conda environment, please check the install guide in PyRosetta website: http://www.pyrosetta.org/dow

## PEPPER

### Basic usage
All methods in PEPPER is wrapped in class named Pepper. Three steps are needed to predict SAVs:
1. Load input file containing SAVs.
2. Scoring the proetin structure energy of these SAVs using REF2015 [2].
3. Predict pathogenicity of input SAVs based on a LightGBM [3] model.

The scripts will look like below:
```python
from pepper import pepper

pepper = pepper.Pepper()
pepper.load_savs('input_sav.txt')
pepper.scoring()
pepper.prediction()
```

After prediction, a csv file named "predict_result.csv" will be generated in folder.
This file contain basic information, predict label, REF2015 score of each SAV.

### Input format
The format of SAVs input file:
> {UniProtKB/Swiss-Prot ID} {Mutate position} {Wildtype amino acid} {Mutated amino acid}

Four elements are seperated by one space and each SAV are seperated into different lines.

It should be like:
```
O95140 263 S P
P08514 273 G D
P02766 136 Y S
...
```
## References
[1] PyRosetta: Chaudhury, Sidhartha, Sergey Lyskov, and Jeffrey J. Gray. "PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta." Bioinformatics 26.5 (2010): 689-691.

[2] REF2015: Alford, Rebecca F., et al. "The Rosetta all-atom energy function for macromolecular modeling and design." Journal of chemical theory and computation 13.6 (2017): 3031-3048.

[3] LightGBM: Ke, Guolin, et al. "Lightgbm: A highly efficient gradient boosting decision tree." Advances in neural information processing systems. 2017.