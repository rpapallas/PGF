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
        #random.shuffle(ids)
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

            #plt.figure(2)
            #plt.plot(history.history['accuracy'], label='training')
            #plt.plot(history.history['val_accuracy'], label='testing')
            #plt.title('Model Accuracy')
            #plt.ylabel('Accuracy')
            #plt.xlabel('# epochs')
            #plt.legend(loc="upper left")
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

            #plt.figure(2)
            #plt.plot(history.history['accuracy'], label='training')
            #plt.plot(history.history['val_accuracy'], label='testing')
            #plt.title('Model Accuracy')
            #plt.ylabel('Accuracy')
            #plt.xlabel('# epochs')
            #plt.legend(loc="upper left")
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
    return HumanPredictor(training_data_size=1500, testing_data_size=20, load_ids_from_file=False, save_ids_to_file=False)

def get_autonomous_predictor():
    return AutonomousPredictor(training_data_size=4000, testing_data_size=500, load_ids_from_file=False, save_ids_to_file=True)

def testing():
    from collections import namedtuple
    Input = namedtuple('Input', 'num_objs dist_to_goal_1 blocking_obst_1 high_force_1 dist_to_goal_2 blocking_obst_2 high_force_2 dist_to_goal_3 blocking_obst_3 high_force_3')
    Output = namedtuple('Output', 'dist_to_goal blocking_obst high_force')
    Test = namedtuple('Test', 'input expected_output')

    MODEL = predictor.get_model()
    TRAINING_DATA, _ = predictor.get_training_data()

    tests = [
        Test(input=Input(num_objs=15, dist_to_goal_1=340.99300000000005, blocking_obst_1=0.0, high_force_1=501.352, dist_to_goal_2=391.89300000000003, blocking_obst_2=450.04699999999997, high_force_2=207.06599999999997, dist_to_goal_3=429.358, blocking_obst_3=450.031, high_force_3=111.34299999999999,), expected_output=Output(dist_to_goal=559.286, blocking_obst=0.0, high_force=75.5641)),
        Test(input=Input(num_objs=10, dist_to_goal_1=336.86400000000003, blocking_obst_1=0.0, high_force_1=0.0, dist_to_goal_2=336.86400000000003, blocking_obst_2=0.0, high_force_2=0.0, dist_to_goal_3=336.86400000000003, blocking_obst_3=0.0, high_force_3=0.0,), expected_output=Output(dist_to_goal=297.372, blocking_obst=0.0, high_force=40.3034)),
        Test(input=Input(num_objs=20, dist_to_goal_1=275.17, blocking_obst_1=0.0, high_force_1=0.0, dist_to_goal_2=275.17, blocking_obst_2=0.0, high_force_2=0.0, dist_to_goal_3=275.17, blocking_obst_3=0.0, high_force_3=0.0,), expected_output=Output(dist_to_goal=0.0528323, blocking_obst=0.0, high_force=0.0)),
        Test(input=Input(num_objs=10, dist_to_goal_1=296.39599999999996, blocking_obst_1=0.0, high_force_1=0.0, dist_to_goal_2=315.948, blocking_obst_2=450.055, high_force_2=0.0, dist_to_goal_3=327.113, blocking_obst_3=0.0, high_force_3=0.0,), expected_output=Output(dist_to_goal=297.318, blocking_obst=0.0, high_force=0.0)),
        Test(input=Input(num_objs=20, dist_to_goal_1=480.36800000000005, blocking_obst_1=450.066, high_force_1=0.0, dist_to_goal_2=484.33099999999996, blocking_obst_2=0.0, high_force_2=36.5981, dist_to_goal_3=484.33099999999996, blocking_obst_3=0.0, high_force_3=36.5981,), expected_output=Output(dist_to_goal=392.606, blocking_obst=0.0, high_force=0.0)),
        Test(input=Input(num_objs=20, dist_to_goal_1=445.86400000000003, blocking_obst_1=450.07099999999997, high_force_1=356.69300000000004, dist_to_goal_2=307.08, blocking_obst_2=0.0, high_force_2=901.4689999999999, dist_to_goal_3=284.228, blocking_obst_3=0.0, high_force_3=593.047,), expected_output=Output(dist_to_goal=402.86300000000006, blocking_obst=0.0, high_force=0.0)),
        Test(input=Input(num_objs=20, dist_to_goal_1=0.0469787, blocking_obst_1=0.0, high_force_1=324.075, dist_to_goal_2=0.0513354, blocking_obst_2=0.0, high_force_2=242.12599999999998, dist_to_goal_3=0.0513354, blocking_obst_3=0.0, high_force_3=242.12599999999998,), expected_output=Output(dist_to_goal=0.0446504, blocking_obst=0.0, high_force=209.80900000000003)),
        Test(input=Input(num_objs=20, dist_to_goal_1=299.482, blocking_obst_1=0.0, high_force_1=242.179, dist_to_goal_2=299.482, blocking_obst_2=0.0, high_force_2=242.179, dist_to_goal_3=329.329, blocking_obst_3=0.0, high_force_3=388.635,), expected_output=Output(dist_to_goal=327.45599999999996, blocking_obst=0.0, high_force=0.0)),
        Test(input=Input(num_objs=20, dist_to_goal_1=666.1030000000001, blocking_obst_1=0.0, high_force_1=281.647, dist_to_goal_2=482.94, blocking_obst_2=450.066, high_force_2=0.0, dist_to_goal_3=489.722, blocking_obst_3=0.0, high_force_3=0.0,), expected_output=Output(dist_to_goal=489.722, blocking_obst=0.0, high_force=0.0)),
        Test(input=Input(num_objs=20, dist_to_goal_1=286.318, blocking_obst_1=0.0, high_force_1=636.484, dist_to_goal_2=286.318, blocking_obst_2=0.0, high_force_2=636.484, dist_to_goal_3=553.68, blocking_obst_3=450.05199999999996, high_force_3=17.4227,), expected_output=Output(dist_to_goal=280.49, blocking_obst=0.0, high_force=0.0)),
    ]
    from sklearn.metrics import mean_squared_error

    for test in tests:
        input_x = np.array([[
            test.input.num_objs,

            test.input.dist_to_goal_1,
            test.input.blocking_obst_1,
            test.input.high_force_1,

            test.input.dist_to_goal_2,
            test.input.blocking_obst_2,
            test.input.high_force_2,

            test.input.dist_to_goal_3,
            test.input.blocking_obst_3,
            test.input.high_force_3,
        ]])

        #xs = np.append(TRAINING_DATA, input_x, axis=0)
        #normalised_xs = tf.keras.utils.normalize(xs, axis=1)
        #input_x = np.array([normalised_xs[-1]])

        y_pred = MODEL.predict(input_x)[0]
        y_actual = test.expected_output.dist_to_goal + test.expected_output.blocking_obst + test.expected_output.high_force
        #rms = mean_squared_error(y_actual, y_pred, squared=False)
        print(y_pred, y_actual)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''\
        TODO: Description
        ''')
    )

    parser.add_argument(
        "--human",
        dest="human_prediction",
        action="store_true",
        help=textwrap.dedent('''\
        Will do human contribution prediction.
        ''')
    )

    args = parser.parse_args()

    predictor = None
    if args.human_prediction:
        print("Predicting for human...")
        predictor = get_human_predictor()
    else:
        print("Predicting for robot...")
        predictor = get_autonomous_predictor()

    predictor.train_and_evaluate_model()
    #testing()
