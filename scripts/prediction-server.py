#!/usr/local/bin/python3
import rospy
import numpy as np
import tensorflow as tf
from pgf.srv import PredictionInput, PredictionInputResponse
from predictors import get_human_predictor, get_autonomous_predictor
import argparse
import textwrap

MODEL = None
NODE_NAME = None
SERVICE_NAME = None
TRAINING_DATA = None

def do_prediction(req):
    input_x = np.array([[#req.numberOfCollisions, 
                  req.numberOfMovableObjects, 

                  req.distanceToGoal[0], 
                  req.blockingObstacleInHand[0], 
                  req.highForce[0], 

                  req.distanceToGoal[1], 
                  req.blockingObstacleInHand[1], 
                  req.highForce[1], 

                  req.distanceToGoal[2], 
                  req.blockingObstacleInHand[2], 
                  req.highForce[2]]])

    y_pred = MODEL.predict(input_x)[0]
    return PredictionInputResponse(predictedCost=y_pred[0])

def gp_server():
    rospy.init_node(NODE_NAME)
    s = rospy.Service(SERVICE_NAME, PredictionInput, do_prediction)
    print(f"[{SERVICE_NAME}] Waiting requests...")
    rospy.spin()

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

    args, unknown = parser.parse_known_args()
    predictor = None

    if args.human_prediction:
        NODE_NAME = "prediction_human"
        SERVICE_NAME = "predict_human"
        print("Predicting for human...")
        predictor = get_human_predictor()
    else:
        NODE_NAME = "prediction_autonomous"
        SERVICE_NAME = "predict_autonomous"
        print("Predicting for robot...")
        predictor = get_autonomous_predictor()

    MODEL = predictor.get_model()
    TRAINING_DATA, _ = predictor.get_training_data()
    gp_server()
