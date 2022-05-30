# UDACITY_MLDevOps

1. Install Anaconda

[Go to](https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/)

2. Update conda

```console
conda update -n base -c defaults conda
```

3. Create environment and install main packages

```console
conda create --name udacity python = 3.8 mlflow jupyter pandas matplotlib requests -c conda-forge
```
4. Activate environment

```console
conda activate udacity
```

5. To turn off the environment

```console
conda deactivate udacity
```
