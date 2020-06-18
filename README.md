# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services

This project showcases the use of unsupervised and supervised machine learning techniques in a real-world application. This project is a capstone project for the Machine Learning Nano Degree by Udacity.

## Project Overview

In this project, I will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. I will use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, I will apply what I&#39;ve learned on a third dataset with demographics information for targets of a marketing campaign for the company and use a model to predict which individuals are most likely to convert into becoming customers for the company.

&nbsp;


**Files Included:**

- Arvato Project Workbook.ipynb: contains the python code for the project
- pytorch\_model.py: defines the parameters for the pytorch model
- model\_utils.py: contains utility methods for the pytorch model
- DIAS Attributes - Values 2017.xlsx: is a top-level list of attributes and descriptions, organized by informational category.
- DIAS Information Levels - Attributes 2017.xlsx: is a detailed mapping of data values for each feature in alphabetical order.
- proposal: the proposal PDF file for the project

&nbsp;

**Libraries used:**

1. numpy
2. pandas
3. matplotlib
4. seaborn
5. sklearn
6. imblearn
7. torch

The code is written in Python 3, Anaconda distribution.

The library imblearn must be installed by opening anaconda terminal and using the following command:

conda install -c conda-forge imbalanced-learn

&nbsp;

**Results Summary:**

After training multiple machine learning models and comparing their results, Random Forest Classifier achieved the best results with ROC AUC score of 0.7.

&nbsp;

**Licenses and Acknowledgements:**

The data for this project was provided by Arvato and cannot be shared publicly.

Special thanks to the mentor Amit L for assisting with the project.

&nbsp;

**Resources:**

Information about the role of AI in advertising:

[https://www.marketingaiinstitute.com/blog/ai-in-advertising](https://www.marketingaiinstitute.com/blog/ai-in-advertising)

[https://econsultancy.com/a-brief-history-of-artificial-intelligence-in-advertising/](https://econsultancy.com/a-brief-history-of-artificial-intelligence-in-advertising/)

The following articles were the main drive behind the chosen supervised learning algorithms:

[https://towardsdatascience.com/pros-and-cons-of-various-classification-ml-algorithms-3b5bfb3c87d6](https://towardsdatascience.com/pros-and-cons-of-various-classification-ml-algorithms-3b5bfb3c87d6)

[https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2264-5](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2264-5)

Other references that assisted in the model refinement process:

[Hyperparameter Tuning the Random Forest in Python - Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

[Batch Normalization and Dropout in Neural Networks with Pytorch](https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd)

[Recall, Precision, F1, ROC, AUC, and everything - The Startup - Medium](https://medium.com/swlh/recall-precision-f1-roc-auc-and-everything-542aedf322b9)

[Scale, Standardize, or Normalize with Scikit-Learn - Towards Data Science](https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02)

[https://imbalanced-learn.readthedocs.io/en/stable/over\_sampling.html](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html)
