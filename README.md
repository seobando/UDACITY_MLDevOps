# UDACITY_MLDevOps

This is the repository for the projects of the Udacity MLDevOps Nanodegree, belog some FAQs:

### How to install anaconda, create and activate a conda environment in Windows?

1. Install anaconda
[Go to](https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/)
2. Open Anaconda Prompt
3. Update conda
```console
conda update -n base -c defaults conda
```
4. Create environment and install main packages
```console
conda create --name udacity python = 3.8 mlflow jupyter pandas matplotlib requests -c conda-forge
```
5. Activate environment
```console
conda activate udacity
```
6. To turn off the environment
```console
conda deactivate udacity
```
### How to install anaconda, create and activate and environment in Windows for Linux Subsystem?

1. Open linux terminal
2. Install anaconda
```console
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
3. Execute it
```console
chmod u+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
```
4. Go to project folder
5. Install environment
```console
conda env create -f environment.yml
```
6. Activate environment
```console
conda activate envName
```
## How to create a wandb account?


### How to run a mlflow step?
```console
mlflow run . -P steps=stepName
```
