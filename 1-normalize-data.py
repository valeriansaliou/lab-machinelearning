#!/usr/bin/env python3

####
# lab-machinelearning
# Valerian Saliou <valerian@valeriansaliou.name>
####

import pandas


class Normalize:
  """
  Handles normalization operations
  """

  def __init__(self, normalize_type, data_path, normalized_path):
    assert data_path

    self.__class = self.__class__.__name__
    self.__normalize_type = normalize_type

    self.__data = pandas.read_csv(data_path)
    self.__normalized_path = normalized_path

  def do(self):
    """
    Proceeds normalization
    """
    print("[%s] Proceeding data normalization..." % self.__normalize_type)
    print()

    # Cleanup
    self.__nuke_features()

    # Normalize
    for feature in self.__data:
      print("[%s] Scan feature: %s" % (self.__normalize_type, feature))

      # Feature should be normalized?
      if ("_%s__normalize_%s" % (self.__class, feature)) in dir(self):
        print("[%s] Normalizing: %s..." % (self.__normalize_type, feature))

        # Proceed normalization
        normalizer = getattr(self, ("_%s__normalize_%s" % (self.__class, feature)))

        normalized_features = normalizer(feature, self.__data[feature])

        if self.__normalize_type == "continuous":
          # Replace normalized column
          self.__data = self.__data.drop(feature, 1)

          for normalized_feature in normalized_features:
            self.__data.insert(0, normalized_feature, normalized_features[normalized_feature])
        elif self.__normalize_type == "categorical":
           self.__data[feature] = normalized_features
        else:
          raise Exception("Type not supported")

    # Output normalized CSV
    self.__data.to_csv(self.__normalized_path, index=False)

    print()
    print("[%s] Proceeded data normalization." % self.__normalize_type)

  def __nuke_features(self):
    """
    Nuke irrelevant features
    """
    nukable_features = [
      "instant",
      "cnt",
      "dteday",
      "atemp",
      "yr"
    ]

    for feature in nukable_features:
      print("[%s] Nuke feature: %s" % (self.__normalize_type, feature))

      self.__data = self.__data.drop(feature, 1)

  def __normalize(self, normalize_map, feature, values):
    """
    Generic normalizer
    """
    if self.__normalize_type == "continuous":
      normalized_table = {}

      # Initialize all normalized columns (to base value, ie: 0)
      for value in normalize_map:
        normalized_table["%s-%s" % (feature, normalize_map[value])] = [0 for i in range(0, len(values))]

      # Translate un-normalized values to normalized values
      for index, value in enumerate(values):
        normalized_table["%s-%s" % (feature, normalize_map[value])][index] = 1

      return normalized_table
    elif self.__normalize_type == "categorical":
      return [normalize_map.get(value, "?") for value in values]
    else:
      raise Exception("Type not supported")

  def __normalize_season(self, feature, values):
    """
    Normalizes season
    """
    normalize_map = {
      1 : "spring",
      2 : "summer",
      3 : "fall",
      4 : "winter"
    }

    return self.__normalize(normalize_map, feature, values)

  def __normalize_mnth(self, feature, values):
    """
    Normalizes mnth
    """
    normalize_map = {
      1  : "january",
      2  : "february",
      3  : "march",
      4  : "april",
      5  : "may",
      6  : "june",
      7  : "july",
      8  : "august",
      9  : "september",
      10 : "october",
      11 : "november",
      12 : "december"
    }

    return self.__normalize(normalize_map, feature, values)

  def __normalize_weekday(self, feature, values):
    """
    Normalizes weekday
    """
    normalize_map = {
      0 : "sunday",
      1 : "monday",
      2 : "tuesday",
      3 : "wednesday",
      4 : "thursday",
      5 : "friday",
      6 : "saturday"
    }

    return self.__normalize(normalize_map, feature, values)

  def __normalize_hr(self, feature, values):
    """
    Normalizes hr
    """
    normalize_map = {
      0  : "0",
      1  : "1",
      2  : "2",
      3  : "3",
      4  : "4",
      5  : "5",
      6  : "6",
      7  : "7",
      8  : "8",
      9  : "9",
      10 : "10",
      11 : "11",
      12 : "12",
      13 : "13",
      14 : "14",
      15 : "15",
      16 : "16",
      17 : "17",
      18 : "18",
      19 : "19",
      20 : "20",
      21 : "21",
      22 : "22",
      23 : "23"
    }

    return self.__normalize(normalize_map, feature, values)

  def __normalize_weathersit(self, feature, values):
    """
    Normalizes weathersit
    """
    normalize_map = {
      1 : "clear",
      2 : "mist",
      3 : "light",
      4 : "heavy"
    }

    return self.__normalize(normalize_map, feature, values)


# Proceed
Normalize("categorical", "./data/hour.csv", "./out/normalized_hour_categorical.csv").do()
Normalize("continuous", "./data/hour.csv", "./out/normalized_hour_continuous.csv").do()
