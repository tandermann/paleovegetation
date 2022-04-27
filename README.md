# The Paleovegetation Prediction Project

Here you can find all code and data associated with the paleovegetation project (predicting past vegetation using mammal, plant, and climate associations). Additional data can be retrieved from the project's [Zenodo repository](https://doi.org/10.5281/zenodo.6492100).

Below you can find a full tutorial for training a Bayesian Neural Network (BNN) model with existing vegetation data, and to use the trained model to predict vegetation through time and space.

# Tutorial

## Building environment

To run this tutorial you will need to install Python as well as several dependencies. We recommend setting up a conda environment to ensure that all dependencies are installed in the correct version. You can do this by following the following instructions:

### Install conda installation manager

Download and install minconda on your computer (unless you already have some conda version installed): https://docs.conda.io/en/latest/miniconda.html.

### Create a conda environment and install software

Now you will create a new environment where the necessary software will be installed. Here we name the enviornment `paleovegetation` (you cna give it any name you like).

```
conda create -n paleovegetation
source activate paleovegetation
conda install python=3.7
python -m pip install https://github.com/dsilvestro/npBNN/archive/refs/tags/0.1.12.tar.gz
```

___
## Set up data and code

1. Download this GitHub repo ([download link](https://github.com/tandermann/paleovegetation/archive/refs/heads/master.zip))
2. Find the downloaded zip-archive in your downloads folder and move it to your working directory (e.g., your Desktop)
3. Unzip the directory
4. Open your command line app (**Mac** and **Linux** users can just use the native Terminal app, but **Windows** users need to make sure to use the conda command prompt that should install together with miniconda)
5. Use `cd` to navigate into the downloaded and unzipped folder
6. All following commands assume that you are located in the downloaded folder

___

## Extract distances and features from raw data

The first step will be to prepare the data for training of our model.

- The raw data includes **vegetation points** (current: `tutorial_data/current_vegetation.txt`, and past: `tutorial_data/paleo_vegetation.txt`). These data consist of spatial coordinates, an associated time (in case of paleovegetation), and the vegetation interpretation (`0` for closed, `1` for open).

- Further, we have coordinates and dates of **fossil occurrences** of selected mammal and plant taxa, as well as available data on their **current occurrences** (`tutorial_data/fossil_and_current_occurrences.txt`).

- Finally, we have data on **additional predictors**, such as global temperature (`tutorial_data/global_average_temperature.txt`), global CO2 content (hard-coded in script `tutorial_1_compile_distances_and_abiotic_features.py`), and elevation (`tutorial_data/elevation`). All of these data are available through time (last 30 million years).

The following script will extract for each vegetation point the distance to the spatially closest fossil occurrence of each taxon for each geological epoch. These distances are part of the input for training our BNN model. Additionally, the script will also export the corresponding temporal distances, as well as extracting additional features (climate, elevation, coordinates, etc.) for each vegetation point. The output will be stored in the folder `tutorial_data/training_data`.

For this tutorial we are only extracting this information for a subset of the available data (all paleovegetation points, but only 331 of the total 11048 of the current vegetation points). Run the script with the following command:

```
python tutorial_1_compile_distances_and_abiotic_features.py
```

___

## Train BNN model

Now we are ready to train a BNN model, using the data we compiled in the previous step. The model will be trained on 80% of the available vegetation data (training set). We will hold back the remaining 20%, which we will use later to make predictions with our trained model, to evaluate how well our model can predict vegetation labels for these data (test set).


The model can be trained by running the following command. Training a BNN model is very time intensive, and training this model until proper convergance of the likelihood will take about 2-3 days of computation time on a "normal" computer. The training is set to run for 500,000 iterations, sampling weights every 200 iterations. With the model configuration we are running here, it takes about 30,000 generations (~2h computation) to reach a reasonable likelihood and prediction accuracy (however, at this point the model is still far from fully converged).

```
python tutorial_2_train_bnn_paleoveg.py
```

If you have the time and patience, let your model training run for a few hours before continuing with the tutorial. Otherwise you can use a pretrained model that is stored at `tutorial/precompiled_data/pretrained_model` for the next steps.

___

## Predict test set with trained model

Now that we have a trained model we can use it to predict the vegetation labels for the 20% of the data which we have been omitting so far. The following script makes predictions for these points and calculates the prediction accuracy of the model. By default the script omits the first 150 posterior samples as burn-in. Therefore make sure that your model has run for more than 30,000 generations (given our sampling frequency of 200 generations). Provide the folder that contains the trained model after the script name (change this to `tutorial/trained_model` to use your own traiend model instead of the precompiled one).

```
python tutorial_3_predict_testset.py tutorial/precompiled_data/pretrained_model
```

The script will print the prediction acccuracy to the screen. In my case the trained model predicted 114 of the 132 test set labels correctly, leading to a 86.4% prediction accuracy:

> Predicted test set.
> 
> Correct predictions: 114 of 132.
> 
> Prediction accuracy: 0.864.


The script also produces an output file (`tutorial/model_predictions/test_set_predictions.txt`) showing the predictions for each test instances, as well as the posterior probability of the predicted labels.

| predicted_vegetation |  true_vegetation | posterior_prob_closed | posterior_prob_open |
| --- | --- | --- | --- |
| 0 | 0 | 0.762 | 0.238 |
| 0 | 0 | 0.857 | 0.143 |
| 0 | 0 | 0.968 | 0.032 |
| 1 | 0 | 0.425 | 0.575 |
| 1 | 1 | 0.144 | 0.856 |
| 1 | 1 | 0.252 | 0.748 |
| 1 | 1 | 0.370 | 0.630 |
| 0 | 0 | 0.940 | 0.060 |
| 0 | 0 | 0.951 | 0.049 |
...
___

## Predict vegetation map with trained model

Just as we predicted the test instances, we can now make predictions for any point in space and time, provided we have extracted spatial and temporal distances and abiotic features for it. We recompiled the distances and abiotic features for all cell-centers of a grid spanning across North America, allowing us to predict the current vegetation map of North America with our trained model. First unzip the folder containing the data: 

```
unzip tutorial/precompiled_data/compiled_data_current_map.zip -d tutorial/precompiled_data/
```

Now run the predictions script, supplying the folder with the folder with the precompiled data after the script name:

```
python tutorial_4_predict_vegetation_map.py tutorial/precompiled_data/compiled_data_current_map
```

The script will plot the predicted vegetation labels for all points, producing a vegetation map (`tutorial/model_predictions/current_map_predicted_labels.pdf`).


Similarly, we can also predict and plot vegetation for points in the past. Here we provide a dataset for predicting the vegetation map of North America 20 million years ago:

```
unzip tutorial/precompiled_data/compiled_data_20MA_map.zip -d tutorial/precompiled_data/
python tutorial_4_predict_vegetation_map.py tutorial/precompiled_data/compiled_data_20MA_map
```

Note that in this tutorial we have only used 80% of the available data, not utilizing a good portion of the precious paleovegetation data we have copmiled for this project. In the published manuscript, we avoided this loss of data by applying the approach of cross-validation, training several models while rotating through all available data. The averaged test accuracies in the manuscript are therefore based on all available data, while reflecting the ability of the modle to predict unseen vegetation labels. We then train one final production model using all data at once and make the predictions with this fully trained model.

Further, in the published manuscript we had access to two additional features for training and prediction, consisting of spatial rasters of temperature and precipitation throughout the last 30 million years, further improving the prediction accuracy and can be requested from the authors. These data were provided to us by Christopher Scotese and can be shared upon request.
