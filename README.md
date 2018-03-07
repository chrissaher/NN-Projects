# NN-Projects
Miscellaneous of different machine-learning related projects. Projects support python 3

## Getting Started
Each project contains it's own **requirements** file for dependencies.

It's highly recommended to use [virtualenv](https://virtualenv.pypa.io/en/stable but not mandatory.

## Projects

- [dinosaur_name_generation](https://github.com/chrissaher/NN-Projects/tree/master/dinosaur_name_generation) - RNN.

- [function_fitter](https://github.com/chrissaher/NN-Projects/tree/master/function_fitter) - Linear regression.

## Run the code
First it's important to install all dependencies

```
pip install -r requirements.txt
```
- Each folder has it's own requirements file in case you only want to run one of them

Then set the PYTHONPATH

```
source .pythonpath
```

Every project contains a **run.sh** file for executing the code.

So for example if you want to run dinosaur_name_generation project, do as follow:

```
source dinosaur_name_generation/run.sh
```
