# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The used model is a Random Forest classifier using the default parameters. The model is used to predict if someone is making more than 50 K in salary based on the following attributes

1. age
2. workclass
3. fnlgt
4. education
5. education-num
6. marital-status
7. occupation
8. relationship
9. race
10. sex
11. capital-gain
12. capital-loss
13. hours-per-week
14. native-country
15. salary

## Intended Use

The model should be used to predict the salary based on a set of attributes. The user of this model are students of the a Udacity Nano Degree Program.

## Training Data

Data can be found here [Census Income data set](https://archive.ics.uci.edu/ml/datasets/census+income) and locally in /data/census.csv

## Evaluation Data

20 % of the data is used for evaluation.

## Metrics
Score: 0.8585905112851221
MAE: 0.14140948871487793
precision: 0.7425018288222385
recall: 0.6407828282828283
fbeta: 0.6879024059640799


## Ethical Considerations

The data used to train the model might contain biases. Some attributes that are collected from conducting a survey, such as hours per week, might contain biases based on influence from friends, co-workers or expectations. Not all countries are represented in the native country attribute and the data set is probably not large enough to assume that the model predicts well with native country attribute.

## Caveats and Recommendations

A very simple Random Forest model with default parameters was trained on the data set and no effort was put into training a model with better performance since the Udacity projects was more about CI/CD. The model would also perform better with a larger data set.