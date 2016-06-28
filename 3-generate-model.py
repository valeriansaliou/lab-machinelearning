#!/usr/bin/env python3 -W ignore::DeprecationWarning

####
# lab-machinelearning
# Valerian Saliou <valerian@valeriansaliou.name>
####

import pandas

from random import random

from plotly.offline import plot
from plotly.graph_objs import Scatter

from sklearn import linear_model, cross_validation
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split


class Model:
  """
  Handles model operations
  """

  def __init__(self, data_path):
    assert data_path

    self.__data = pandas.read_csv(data_path)

    self.__size_train = 0.6
    self.__best_model_test_trials = 10

  def generate(self):
    """
    Generates the model
    """
    print("Generating model...")
    print()

    # Split the input data in 2 sets (train + test)
    train_set, test_set = self.__split_data()

    # Pick relevant columns
    predict_columns = ["casual", "registered"]

    train_set_x = train_set.drop(predict_columns, axis=1)
    train_set_y = pandas.DataFrame(train_set, columns=predict_columns)

    test_set_x = test_set.drop(predict_columns, axis=1)
    test_set_y = pandas.DataFrame(test_set, columns=predict_columns)

    # Proceed linear regression on test models
    for model_name in ["linear", "elasticnet", "lars"]:
      model = self.__pick_best_model_instance(model_name, train_set_x, train_set_y, test_set_x, test_set_y)

      # Plot prediction performance
      self.__plot_prediction(model_name, model, [
        ["Train", train_set_x, train_set_y],
        ["Test", test_set_x, test_set_y]
      ])

      # Process cross-validation
      # Default is 5-fold cross-validation, but cross-validation on the Lars \
      #   model makes the score go below zero. Thus, proceed a 10-fold \
      #   cross-validation for the Lars model to fix it.
      fold_hardfix = 10 if model_name == "lars" else 5

      scores = cross_validation.cross_val_score(model, train_set_x, train_set_y, cv=fold_hardfix)

      print()
      print("[%s] Score by fold: %s" % (model_name, scores))
      print("[%s] Accuracy: %0.4f (+/- %0.2f)" % (model_name, scores.mean(), scores.std() * 2))

    print()
    print("Done generating model.")

  def __split_data(self):
    """
    Splits the data
    """
    return train_test_split(self.__data, train_size=self.__size_train, random_state=0)

  def __pick_best_model_instance(self, model_name, train_set_x, train_set_y, test_set_x, test_set_y):
    """
    Picks the best model instance
    """
    best_model = None
    highest_score = 0.0

    # Iterate on possible model values
    for model_instance, model_hyperparameters in self.__build_model_instances(model_name):
      model_instance.fit(train_set_x, train_set_y)

      model_test_reference = test_set_y.values
      model_test_predicted = model_instance.predict(test_set_x)

      # Process R2 score
      score = r2_score(model_test_reference, model_test_predicted)

      if score > highest_score:
        best_model = model_instance
        highest_score = score

    print("[%s] R2 score: %s" % (model_name, highest_score))
    print("[%s] Hyper-parameters: %s" % (model_name, model_hyperparameters))

    return best_model

  def __build_model_instances(self, model_name):
    """
    Builds model instances
    """
    if model_name == "linear":
      test_iterator = self.__best_model_test_linear()
    elif model_name == "elasticnet":
      test_iterator = self.__best_model_test_elasticnet()
    elif model_name == "lars":
      test_iterator = self.__best_model_test_lars()
    else:
      raise Exception("Model not supported")

    return test_iterator

  def __best_model_test_linear(self):
    """
    Acquire best model test hyper-parameters for linear regression
    """
    yield linear_model.LinearRegression(), []

  def __best_model_test_elasticnet(self):
    """
    Acquire best model test hyper-parameters for elasticnet regression
    """
    # Format: [alpha, l1_ratio]
    ranges = [
      (0.01, 1.0),
      (1.0, 0.01)
    ]

    for test_data in self.__generate_trial_hyperparameters(ranges):
      yield linear_model.ElasticNet(alpha=test_data[0], l1_ratio=test_data[1]), test_data

  def __best_model_test_lars(self):
    """
    Acquire best model test hyper-parameters for lars regression
    """
    yield linear_model.Lars(), []

  def __generate_trial_hyperparameters(self, ranges, hyperparameters=[]):
    """
    Generates trial hyper-parameters
    """
    # Trials storage
    trials = range(0, self.__best_model_test_trials)

    # Initialize?
    if len(hyperparameters) is 0:
      hyperparameters = [[] for i in trials]

    # Generate for first range available
    current_range = ranges.pop(0)
    current_range_span = ((current_range[1] - current_range[0]) / (self.__best_model_test_trials - 1))

    for i in trials:
      # Generate {i}th number in range
      hyperparameters[i].append(current_range[0] + (i * current_range_span))

    # Descend to deeper range?
    if len(ranges) >= 1:
      self.__generate_trial_hyperparameters(ranges, hyperparameters=hyperparameters)

    return hyperparameters

  def __plot_prediction(self, model_name, model, sets):
    """
    Plots prediction
    """
    scatter_items = []

    for plot_set in sets:
      set_predict = model.predict(plot_set[1])
      set_reference = plot_set[2].values

      residual_set = (set_predict - set_reference)

      scatter_items.append(
        Scatter(
          x=residual_set[:, 0],
          y=residual_set[:, 1],
          mode="markers",
          name=plot_set[0]
        )
      )

    # Print a scatterplot of the quality of prediction
    plot(scatter_items, filename=("./out/model_%s_test.html" % model_name))


# Proceed
model = Model("./out/normalized_hour_continuous.csv")

model.generate()
