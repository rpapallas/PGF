import random
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import argparse
import textwrap
from os import path
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental import preprocessing
from collections import namedtuple
import getpass

PLOT = True

DATA_PATH = os.path.expanduser("~/data")

class Predictor:
    def __init__(self, training_data_size, testing_data_size, load_ids_from_file=False, save_ids_to_file=False):
        self.load_ids_from_file = load_ids_from_file
        self.save_ids_to_file = save_ids_to_file
        self.training_data_size = training_data_size
        self.testing_data_size = testing_data_size

        self.df = self._get_df()
        self.training_ids, self.testing_ids = self._get_ids()

    def train_and_evaluate_model(self):
        model = self.get_model()
        x_test, y_test = self.get_testing_data()
        model.evaluate(x_test, y_test, verbose=1)

    def _get_df(self):
        df1 = pd.read_csv(self.data_path + "/runs.txt", sep=", ", engine='python')
        df2 = pd.read_csv(self.data_path + "/experiments.txt", sep=", ", engine='python')
        return pd.merge(df1, df2, left_on='experiment_id', right_on='experiment_id', how='left')

    def _get_object_position_df(self):
        return pd.read_csv(self.data_path + "/object_positions.txt", sep=", ", engine='python')

    def _get_data_from_ids(self, ids):
        mask = self.df["experiment_id"].isin(ids)
        data = self.df[mask]
        return data.groupby("experiment_id", as_index=False)

    def _get_object_collisions_df(self):
        df = pd.read_csv(self.data_path + "/distances.txt", sep=", ", engine='python')
        return df

    def _save_shuffled_ids_to_file(self, ids):
        with open(self.data_path + "/ids.txt", "w") as ids_file:
            for idx in ids:
                ids_file.write(str(idx) + "\n")

    def _get_random_ids(self):
        ids = list(range(1, self.training_data_size + self.testing_data_size + 1))
        random.shuffle(ids)
        return ids

    def _load_ids_from_file(self):
        ids = []
        with open(self.data_path + "/ids.txt") as ids_file:
            lines = ids_file.readlines()
            ids = [int(idx.rstrip()) for idx in lines]

        return ids

    def get_training_data(self):
        training_data = self._get_data_from_ids(self.training_ids)
        x_train, y_train = self._get_data(training_data)
        return x_train, y_train

    def get_testing_data(self):
        testing_data = self._get_data_from_ids(self.testing_ids)
        x_test, y_test = self._get_data(testing_data)
        return x_test, y_test

    def _get_value(self, df, index, column_name):
        return df.iloc[index, df.columns.get_loc(column_name)]

    def _root_mean_squared_error(self, y_true, y_pred):
        return tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred))

    def _get_ids(self):
        ids = None
        if self.load_ids_from_file:
            ids = self._load_ids_from_file()
        else:
            ids = self._get_random_ids()
            if self.save_ids_to_file:
                self._save_shuffled_ids_to_file(ids)

        training_ids = ids[:self.training_data_size]
        testing_ids = ids[self.training_data_size:self.training_data_size + self.testing_data_size:]

        return training_ids, testing_ids

class AutonomousPredictor(Predictor):
    def __init__(self, training_data_size, testing_data_size, load_ids_from_file=False, save_ids_to_file=False):
        self.data_path = DATA_PATH
        super().__init__(training_data_size, testing_data_size, load_ids_from_file, save_ids_to_file)

    def get_model(self):
        training_ids, _ = self._get_ids()

        x_train, y_train= self.get_training_data()
        normalizer = preprocessing.Normalization()
        normalizer.adapt(x_train)

        model = tf.keras.models.Sequential([
          normalizer,
          tf.keras.layers.Dense(64, activation=tf.nn.relu),
          tf.keras.layers.Dense(32, activation=tf.nn.relu),
          tf.keras.layers.Dense(16, activation=tf.nn.relu),
          tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='Adam', loss=self._root_mean_squared_error)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=50, shuffle=True, callbacks=[es_callback])

        if PLOT:
            plt.figure(1)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='testing')
            plt.title('Model Error')
            plt.ylabel('RMSE')
            plt.xlabel('# epochs')
            plt.legend(loc="upper left")

            plt.show()

        return model

    def _get_data(self, data):
        xs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        ys = np.array([[0]])
        look_ahead = 10

        Point = namedtuple("Point", "distance_to_goal blocking_obstalce_in_hand high_force")

        for experiment_id, run in data:
            scene_name = run["scene_name"].iloc[0]
            num_of_collisions = run["number_of_collisions"].iloc[0]
            number_of_movable_objects = run["number_of_movable_obstacles"].iloc[0]

            reaching_costs = {}
            index = 0
            for i in range(len(run)):
                current_distance_to_goal = self._get_value(run, i, "DistanceToGoal")
                current_blocking_obstacle_in_hand = self._get_value(run, i, "BlockingObstacleInHand")
                current_high_force = self._get_value(run, i, "HighForce")
                point = Point(distance_to_goal=current_distance_to_goal, blocking_obstalce_in_hand=current_blocking_obstacle_in_hand, high_force=current_high_force)
                reaching_costs[i] = point

                desired_start_index = i - look_ahead - 3
                index1 = desired_start_index
                index2 = desired_start_index + 1
                index3 = desired_start_index + 2
                indices_in_reaching_costs = [index in reaching_costs for index in [index1, index2, index3]]
                if all(indices_in_reaching_costs):
                    number_of_movable_objects = run["number_of_movable_obstacles"].iloc[i]
                    xs = np.append(xs,
                        np.array([[
                            number_of_movable_objects,

                            reaching_costs[index1].distance_to_goal,
                            reaching_costs[index1].blocking_obstalce_in_hand,
                            reaching_costs[index1].high_force,

                            reaching_costs[index2].distance_to_goal,
                            reaching_costs[index2].blocking_obstalce_in_hand,
                            reaching_costs[index2].high_force,

                            reaching_costs[index3].distance_to_goal,
                            reaching_costs[index3].blocking_obstalce_in_hand,
                            reaching_costs[index3].high_force,
                        ]]), axis=0)

                    future_distance_to_goal = self._get_value(run, i, "DistanceToGoal")
                    future_blocking_obstacle_in_hand = self._get_value(run, i, "BlockingObstacleInHand")
                    future_high_force = self._get_value(run, i, "HighForce")

                    last_cost = [future_distance_to_goal + future_blocking_obstacle_in_hand + future_high_force]
                    ys = np.append(ys, np.array([last_cost]), axis=0)

        xs = xs[1:]
        ys = ys[1:]
        return xs, ys

class HumanPredictor(Predictor):
    def __init__(self, training_data_size, testing_data_size, load_ids_from_file=False, save_ids_to_file=False):
        self.data_path = f"{DATA_PATH}/human"
        super().__init__(training_data_size, testing_data_size, load_ids_from_file, save_ids_to_file)

    def get_model(self):
        training_ids, _ = self._get_ids()

        x_train, y_train= self.get_training_data()
        normalizer = preprocessing.Normalization()
        normalizer.adapt(x_train)

        model = tf.keras.models.Sequential([
          normalizer,
          tf.keras.layers.Dense(64, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(32, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(16, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.optimizers.Adam(), loss=self._root_mean_squared_error)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        history = model.fit(x_train, y_train, validation_split=0.3, epochs=100, batch_size=50, shuffle=True, callbacks=[es_callback])

        if PLOT:
            plt.figure(1)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='testing')
            plt.title('Model Error')
            plt.ylabel('RMSE')
            plt.xlabel('# epochs')
            plt.legend(loc="upper left")

            plt.show()

        return model

    def get_average(self):
        mask = self.df["experiment_id"].isin(self.training_ids)
        data = self.df[mask]
        data = data.groupby("experiment_id", as_index=False)
        look_ahead = 10

        distance_to_goal = []
        blocking_obstalce_in_hand = []
        high_force = []

        for _, run in data:
            latest_distance_to_goal = 0
            latest_blocking_obstacle_in_hand = 0
            latest_high_force = 0

            for i in range(len(run)):
                if run["type"].iloc[i] == "reaching":
                    latest_distance_to_goal = self._get_value(run, i, "DistanceToGoal")
                    latest_blocking_obstacle_in_hand = self._get_value(run, i, "BlockingObstacleInHand")
                    latest_high_force = self._get_value(run, i, "HighForce")
                if run["type"].iloc[i] == "reaching_after_human_input":
                    if (len(run) - 1) < (i + look_ahead):
                        future_distance_to_goal =  self._get_value(run, len(run) - 1, "DistanceToGoal")
                        future_blocking_obstacle_in_hand = self._get_value(run, len(run) - 1, "BlockingObstacleInHand")
                        future_high_force = self._get_value(run, len(run) - 1, "HighForce")
                    else:
                        future_distance_to_goal = self._get_value(run, i + look_ahead, "DistanceToGoal")
                        future_blocking_obstacle_in_hand = self._get_value(run, i + look_ahead, "BlockingObstacleInHand")
                        future_high_force = self._get_value(run, i + look_ahead, "HighForce")

                    distance_to_goal.append(future_distance_to_goal)
                    blocking_obstalce_in_hand.append(future_blocking_obstacle_in_hand)
                    high_force.append(future_high_force)

        return np.mean(distance_to_goal), np.mean(blocking_obstalce_in_hand), np.mean(high_force)

    def _get_data(self, data):
        xs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        ys = np.array([[0]])
        
        look_ahead = 10

        Point = namedtuple("Point", "distance_to_goal blocking_obstalce_in_hand high_force")

        for experiment_id, run in data:
            scene_name = run["scene_name"].iloc[0]
            num_of_collisions = run["number_of_collisions"].iloc[0]
            number_of_movable_objects = run["number_of_movable_obstacles"].iloc[0]

            reaching_costs = {}
            index = 0
            last_cost = None
            for i in range(len(run)):
                if run["type"].iloc[i] == "reaching":
                    current_distance_to_goal = self._get_value(run, i, "DistanceToGoal")
                    current_blocking_obstacle_in_hand = self._get_value(run, i, "BlockingObstacleInHand")
                    current_high_force = self._get_value(run, i, "HighForce")
                    point = Point(distance_to_goal=current_distance_to_goal, blocking_obstalce_in_hand=current_blocking_obstacle_in_hand, high_force=current_high_force)
                    reaching_costs[i] = point
                elif run["type"].iloc[i] == "pushing":
                    index += 1
                elif run["type"].iloc[i] == "reaching_after_human_input":
                    actual_look_ahead_iteration = i - index
                    desired_start_index = actual_look_ahead_iteration - look_ahead - 3
                    index1 = desired_start_index
                    index2 = desired_start_index + 1
                    index3 = desired_start_index + 2
                    indices_in_reaching_costs = [index in reaching_costs for index in [index1, index2, index3]]
                    if all(indices_in_reaching_costs):
                        number_of_movable_objects = run["number_of_movable_obstacles"].iloc[i]
                        xs = np.append(xs,
                            np.array([[
                                number_of_movable_objects,

                                reaching_costs[index1].distance_to_goal,
                                reaching_costs[index1].blocking_obstalce_in_hand,
                                reaching_costs[index1].high_force,

                                reaching_costs[index2].distance_to_goal,
                                reaching_costs[index2].blocking_obstalce_in_hand,
                                reaching_costs[index2].high_force,

                                reaching_costs[index3].distance_to_goal,
                                reaching_costs[index3].blocking_obstalce_in_hand,
                                reaching_costs[index3].high_force,
                            ]]), axis=0)

                        future_distance_to_goal = self._get_value(run, i, "DistanceToGoal")
                        future_blocking_obstacle_in_hand = self._get_value(run, i, "BlockingObstacleInHand")
                        future_high_force = self._get_value(run, i, "HighForce")

                        last_cost = [future_distance_to_goal + future_blocking_obstacle_in_hand + future_high_force]
                        ys = np.append(ys, np.array([last_cost]), axis=0)

        xs = xs[1:]
        ys = ys[1:]
        return xs, ys

def get_human_predictor():
    return HumanPredictor(training_data_size=1500, testing_data_size=500, load_ids_from_file=False, save_ids_to_file=False)

def get_autonomous_predictor():
    return AutonomousPredictor(training_data_size=4000, testing_data_size=500, load_ids_from_file=False, save_ids_to_file=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--human",
        dest="human_prediction",
        action="store_true")

    args = parser.parse_args()

    predictor = None
    if args.human_prediction:
        print("Predicting for human...")
        predictor = get_human_predictor()
    else:
        print("Predicting for robot...")
        predictor = get_autonomous_predictor()

    predictor.train_and_evaluate_model()
