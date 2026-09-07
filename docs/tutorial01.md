# Tutorial

curvelab is program that lets one load, modify, plot, and fit XY data sets. Data can be loaded from a variety of formats, plotting is done leveraging matplotlib, and fittings makes use of lmfit.

This tutorial covers:

+ loading the data from a CSV file
+ plotting the selected columns
+ creating a fitting session and creating a model
+ fitting the initial model and deciding whether it is a good representation of the data
+ cleaning the data and creating a better model
+ fitting with and without errors

## Loading and plotting the data

We use the data from the file `tutorial_two_peaks.csv`. The model comprises two Gaussians and a line: noise and one outlier were added, so to be able to illustrate some of the features of curvelab.

We start by loading the data and adding a series

![](image_1.png)

![](image_2.png)

We can then plot the series

![](image_3.png)

## Creating a fitting session

We now create a fitting session called 'Tutorial model 1', with a model that is the sum of a Gaussian and a Linear component. We can click on Auto Guess to have lmfit get the initial values for the fit.

![](image_4.png)

Then we can click on Fit to have lmfit perform the fit. The model is plotted with the data.

![](image_5.png)

curvelab now shows the fitted parameter values calculated by lmfit, and the complete fit report is shown in the panel at the bottom.

The quality of the fit can be judged from the statistics returned by lmfit:
+ the reduced chi-squared
+ the Akaike Information Criterion statistic
+ the Bayesian Information Criterion statistic

