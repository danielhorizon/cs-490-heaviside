# flexible-metric-optimization

This repository demonstrates a new method of training a machine learning classifier by directly optimizing for confusion-matrix-based measurements. This is accomplished using a continuous approximation of the step function to make the final metric differentiable. The code for this method and for loss functions that incorporate it is included in this repository, along with a set up to test it on various datasets.

The jupyter notebook [`Paper Plots.ipynb`](notebooks/Paper%20Plots.ipynb) already has the results from our original experiments on this training technique. These results show how the model scored in various performance metrics when trained using our method, as well as how it performed with other known training methods. The notebook also has illustrations of our step function approximation.

If you would like to run your own experiment, just follow the instructions below. After completing your test runs, use the notebook [as specified](https://github.com/yale-img/flexible-metric-optimization#3-analyzing-the-results) to observe the final results.

## Downloading files and dependencies

### Step 0:
[Install docker](https://docs.docker.com/engine/install/) (if you don't have it):

### Step 1: Clone the Repository:
If you haven't cloned this repository already, cd into the location where you wish to store this repository and enter:  
    <br>
    `git clone https://github.com/yale-img/flexible-metric-optimization`

### Step 2: Download the test datasets
cd into the project folder:  
    <br>
    `cd flexible-metric-optimization`  
    <br>
Then run  
    <br>
    `./scripts/download_data.sh`  
    <br>
This will download three of the datasets which were used in the original experiments: the kaggle fraud detection dataset, and the uci adult dataset, and the mammography dataset. (To download the fourth dataset that we used, follow [these instructions](https://github.com/yale-img/flexible-metric-optimization/blob/master/README.md#downloading-the-cocktailparty-dataset).) The parsing code for all of these datasets in in `src/datasets.py`.


## Setting up a docker container and running an experiment
Next, set up the docker container and collect all the dependencies for the repository

### 1. For running the tensorflow programs:  
1. cd into the folder (if you haven't) and run:  
    <br>
    `yarn build` (You only need to use this command once in the project's folder)  
    `yarn start` (You also need this just once, unless you have run `yarn stop` at some point)  
    `yarn shell`  
    <br>
This will bring you into the container.  
Then cd into the main folder:  
    <br>
    `cd ~/timeseries`  
    <br>

1. From here you can run an experiment by specifying any of the loss functions you want to test (all defined in losses.py), any of the given datasets you want to try it on (all preprocessed in datasets.py), and other parameters such as the training batch size, number of trials of the experiment, and number of training epochs.  
    `./src/main.py --loss <desired loss> --dataset <desired dataset> --experiment <experiment folder> [--trials <number of trials>] [--nprocs <number of threads>] [--batch_size <batch size>] [--epchs <number of epochs>] ...`  
For example:  
    <br>
    `./src/main.py --experiment t52 --trials 10 --nprocs 10 --loss f1_05 --dataset synthetic --batch_size 512`
    <br>  
This command will train the model on datasets listed in the "synthetic" category, by optimizing for the f1 score where the threshold used in the step function approximation is 0.5. The results are saved in an experiment folder "t52", in `experiments/running/t52`  
<br>

### 2. Running the pytorch experiments:
1. cd into the main directory, then run the following:  
    <br>
    `yarn torch-build`  
    `yarn torch-start`  
    `yarn torch-shell`  
    <br>
These work similarty to yarn build, yarn start, and yarn shell. Once in the container, cd into the main directory:  <br>
    `cd ~/timeseries`   

2. Once you're in the container, you can run an experiment as follows:  
    `./src/torchbcemain.py --loss <loss function> --dataset <dataset name> --mode <train/test> --batch_size <batch size> --experiment <experiment folder>`  
For example:  
    <br>
    `./src/torchbcemain.py --loss approx-f1 --dataset kaggle_cc_fraud --mode train --batch_size 20 --experiment TORCH5`  
    <br>
This trains the model on the kaggle_cc_fraud dataset by optimizing for the f1 score based on the proposed heaviside approximation, and saves the results in a folder called "TORCH5".  

### Note:
The duration of these experiments can range from a few minutes to a few hours. Allow an experiment to finish to ensure that you have the complete set of output data for analysis. If you're forced to stop the process or want to rerun a completed experiment for some reason, just execute the same experiment by adding the argument `--overwrite` at the end of your run command. This will start a new record all over again.

### 3. Analyzing the results:  
The next step is to analyze the files generated during the train/test process using a jupyter notebook. The notebook generates tables and graphs indicating how the model performed after running on the specified loss(es) and as measured by various metrics.  
1. Open the jupyter notebook `Paper Plots.ipynb` found in the "notebooks" folder.  
1. Sections 2 through 4 of the notebook (labeled as Experiment 1, 2 and 3) analyze the files generated by running `main.py`. To run any of these sections, you must add the experiment folder of the results you want to analyze in the first cell of that section.  
For example, for section 2 (Experiment 1), if you're analyzing the results in a folder called 'foo', rewrite the value assignment for the EXP1 variable as follows:  
    `EXP1 = 'foo'`  
1. Section 5 (labeled as Experiment 4) obtains the end results from files generated by running torchbcemain.py. Here too, you assign the name of the experiment's output folder to the variable EXP4 in the first cell of the section.  


### Downloading the CocktailParty Dataset

We use a preprocessed version of the cocktailparty dataset that is suited for binary classification. To obtain this version run `binary_cocktail_party.ipynb` jupyter notebook on your browser. It is included in the `notebooks/` folder and it has the instructions and download links to the original datasets. The notebook will generate the preprocessed version you need.
