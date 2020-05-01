# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# pip install pandas etc
# installation
# library(reticulate)
# py_install("pandas", "numpy", "matplotlib", "sklearn")


# Data Management
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# topics to study
#chapter 5, statistics, c
#chapter 6, probability - dependence and independence
#chapter 6,  probability - conditional probability
#chapter 6,  probability - Bayes’s Theorem
#chapter 6,  probability - Continuous Distributions
#chapter 6,  probability - The Normal Distribution
#chapter 6,  probability - The Central Limit Theorem
#Chapter 7. Hypothesis and Inference
# Statistical Hypothesis Testing
# Confidence Intervals
# P-hacking
# Bayesian Inference

# Chapter 11. Machine Learning
# Before we can talk about machine learning we need to talk about models.
# What is a model? It’s simply a specification of a mathematical (or probabilistic)
# relationship that exists between different variables.
# creating and using models that are learned from data

# might be called predictive modeling or data mining
# but we will stick with machine learning
# use existing data to develop models that we can use to predict various
# outcomes for new data, such as

# We’ll look at both supervised models (in which there is a set of data labeled with the
# correct answers to learn from), and unsupervised models (in which there are no such
# labels).

# we might assume that a person’s height is (roughly) a linear function of his
# weight and then use data to learn what that linear function is

# Or we might assume that a
# decision tree is a good way to diagnose what diseases our patients have and then use data
# to learn the “optimal” such tree.

# A common danger in machine learning is overfitting — producing a model that performs
# well on the data you train it on but that generalizes poorly to any new data. This could
# involve learning noise in the data. Or it could involve learning to identify specific inputs
# rather than whatever factors are actually predictive for the desired output.
# 
# The other side of this is underfitting, producing a model that doesn’t perform well even on
# the training data, although typically when this happens you decide your model isn’t good
# enough and keep looking for a better one.
#######################################################################################
# Clearly models that are too complex lead to overfitting and don’t generalize well beyond
# the data they were trained on. So how do we make sure our models aren’t too complex?
# The most fundamental approach involves using different data to train the model and to test
# the model.
##########################################################################################

#1split your data set, so that (for example) two-thirds of it is
# used to train the model, after which we measure the model’s performance on the
# remaining third





