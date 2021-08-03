//  Copyright (C) 2019 Rafael Papallas and The University of Leeds
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//  Author: Rafael Papallas (rpapallas.com)

#include <boost/optional.hpp>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include "Execution.cpp"
#include "pgf/RobotBid.h"
#include "pgf/HumanRelease.h"
#include "../src/discrete/StraightLineOptimizer.cpp"
#include "../utils/CommonFunctions.cpp"

const string OPERATOR_NAME = "Rafael";
string ROBOT_NAME;
extern string EXPERIMENT_PATH;
extern bool isZoomLocked;
const string PLANNER_NAME = "pgf";
vector<string> STATIC_OBSTACLE_NAMES = {"shelf"};
vector<string> MOVABLE_OBSTACLE_NAMES;
int NUMBER_OF_MOVABLE_OBJECTS;

// ================================================================
// GENERAL PARAMETERS
// ================================================================
const double PLANE_HIGH_X = 0.58;
const double PLANE_LOW_X = 0.20;
const double PLANE_HIGH_Y = 0.38;
const double PLANE_LOW_Y = -0.38;
const double TABLE_Z = 0.18;

// ================================================================
// TRAJECTORY OPTIMISATION PARAMETERS FOR PUSHING
// ================================================================
string SCENE_NAME;
const string GOAL_OBJECT_NAME = "object_3";
int TRAJECTORY_ROLLOUTS = 8;
double COST_THRESHOLD = 20.0;
int TRAJECTORY_DURATION = 3;
int CONTROL_SEQUENCE_STEPS = 8;
vector<double> VARIANCE_VECTOR = {0.04, 0.04, 0.04};

// Parameters for experiments
bool ADAPTIVE = true;
bool ADAPTIVE_LOCAL_MINIMA = true;
int NUMBER_OF_CONSECUTIVE_NON_DECREASES = 5;

bool PREDICTIVE = true; // TensorFlow-based prediction
float THRESHOLD_PREDICTION = 1;

//double TIMEOUT_FOR_HELP = 30.0;
double TIMEOUT_FOR_HELP = 180.0;
double TOTAL_TIME_LIMIT = 180.0;

double INTERACTION_TIME = 0.0;
bool plannerIsIdle = false;
double totalIdleTime = 0.0;
boost::optional<std::chrono::high_resolution_clock::time_point> humanReturnedStartClock;
boost::optional<std::chrono::high_resolution_clock::time_point> humanReturnedWhilePlannerIdleStartClock;
std::chrono::high_resolution_clock::time_point plannerWaitingStartClock;

double formatDurationToSeconds(std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end) {
    int milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    return milliseconds / 1000.0;
}

void window_focus_callback(GLFWwindow* /*window*/, int windowIsFocused) {
    if(windowIsFocused) {
        humanReturnedStartClock = std::chrono::high_resolution_clock::now();
        if(plannerIsIdle) {
            humanReturnedWhilePlannerIdleStartClock = humanReturnedStartClock;
            totalIdleTime += formatDurationToSeconds(plannerWaitingStartClock, humanReturnedStartClock.get());
        } else {
            humanReturnedWhilePlannerIdleStartClock = boost::none;
        }
    } else {
        if (plannerIsIdle) {
            plannerWaitingStartClock = std::chrono::high_resolution_clock::now();
        }

        if(humanReturnedWhilePlannerIdleStartClock && plannerIsIdle) {
            auto now = std::chrono::high_resolution_clock::now();
            INTERACTION_TIME += formatDurationToSeconds(humanReturnedWhilePlannerIdleStartClock.get(), now);
        }

        humanReturnedStartClock = boost::none;
    }
}

int getNextId(const string &filename) {
    ifstream inputFile(EXPERIMENT_PATH + filename);
    int experimentNumber;
    inputFile >> experimentNumber;
    return experimentNumber;
}

int getNextActionId() {
    return getNextId("next_action_id.txt");
}

int getNextExperimentId() {
    return getNextId("next_experiment_id.txt");
}

void updateNextId(const string &filename, int id) {
    id++;
    std::ofstream outfile;
    outfile.open(EXPERIMENT_PATH + filename);
    outfile << id << "\n";
}

void updateNextExperimentId(int experimentId) {
    updateNextId("next_experiment_id.txt", experimentId);
}

void updateNextActionId(int actionId) {
    updateNextId("next_action_id.txt", actionId);
}

void writeAction(int experimentId, int actionId, const string& actionType, string outcome, double optimizationTime, double interactionTime) {
    std::ofstream outfile;
    outfile.open(EXPERIMENT_PATH + "actions.txt", std::ios_base::app);

    if(actionType == "reach")
        interactionTime = 0.0;

    outfile << experimentId
        << ", "
        << actionId
        << ", "
        << actionType
        << ", "
        << outcome
        << ", "
        << optimizationTime
        << ", "
        << interactionTime
        << "\n";
}

void writeExperiment(int experimentId, const string& outcome, const string& updateType, double timeToSolve, double totalTimeWaitingForHuman, double totalTimeWastedWaitingForHuman) {
    std::ofstream outfile;
    outfile.open(EXPERIMENT_PATH + "experiments.txt", std::ios_base::app);

    string timeout;
    if(ADAPTIVE && ADAPTIVE_LOCAL_MINIMA)
        timeout = "adaptive_local_minima";
    else if(ADAPTIVE)
        timeout = "adaptive_" + to_string(NUMBER_OF_CONSECUTIVE_NON_DECREASES);
    else if(PREDICTIVE)
        timeout = "predictive";
    else if(TIMEOUT_FOR_HELP == TOTAL_TIME_LIMIT)
        timeout = "no_timeout";
    else
        timeout = to_string(TIMEOUT_FOR_HELP);

    string thresholdPrediction = "N/A";
    if(PREDICTIVE)
        thresholdPrediction = to_string(THRESHOLD_PREDICTION);

    time_t now = time(0);
    char* datetime = ctime(&now);

    outfile << experimentId
        << ", "
        << PLANNER_NAME
        << ", "
        << SCENE_NAME
        << ", "
        << NUMBER_OF_MOVABLE_OBJECTS
        << ", "
        << OPERATOR_NAME
        << ", "
        << ROBOT_NAME
        << ", "
        << outcome
        << ", "
        << TOTAL_TIME_LIMIT
        << ", "
        << timeout
        << ", "
        << TRAJECTORY_ROLLOUTS
        << ", "
        << COST_THRESHOLD
        << ", "
        << TRAJECTORY_DURATION
        << ", "
        << CONTROL_SEQUENCE_STEPS
        << ", "
        << VARIANCE_VECTOR[0]
        << ", "
        << VARIANCE_VECTOR[1]
        << ", "
        << VARIANCE_VECTOR[2]
        << ", "
        << updateType
	<< ", "
	<< timeToSolve
	<< ", "
	<< THRESHOLD_PREDICTION
	<< ", "
	<< totalTimeWaitingForHuman
	<< ", "
	<< totalTimeWastedWaitingForHuman
	<< ", "
	<< datetime
        << "\n";
}

void writeCostSequence(int actionid, double cost) {
    std::ofstream outfile;
    outfile.open(EXPERIMENT_PATH + "cost_sequence.txt", std::ios_base::app);

    outfile << actionid
        << ", "
        << cost
        << "\n";
}

void moveBack() {
    finishedExecution = false;
    auto solution = moveRobotBackWithTrajectoryOptimization(MOVABLE_OBSTACLE_NAMES, STATIC_OBSTACLE_NAMES, PLANE_HIGH_X, PLANE_LOW_X, PLANE_HIGH_Y, PLANE_LOW_Y, TABLE_Z);

    auto optimalControlSequence = get<0>(solution);
    auto actionDuration = get<1>(solution);

    double steps = actionDuration / globalMujocoHelper->getTimeStep();

    for(auto control : optimalControlSequence.getControls()) {
        for (int step = 0; step < steps; ++step) {
            globalMujocoHelper->setRobotVelocity(control.getLinearX(), control.getLinearY(), control.getAngularZ());
            globalMujocoHelper->step();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    finishedExecution = true;
}

void setupOptimizer(StraightLineInteractiveOptimizer &optimizer) {
    optimizer.setNumberOfNoisyTrajectoryRollouts(TRAJECTORY_ROLLOUTS);
    optimizer.setTrajectoryDuration(TRAJECTORY_DURATION);
    optimizer.setSamplingVarianceVector(&VARIANCE_VECTOR);
    optimizer.setStaticObstacleNames(&STATIC_OBSTACLE_NAMES);
    optimizer.setMovableObstacleNames(&MOVABLE_OBSTACLE_NAMES);
    optimizer.setControlSequenceSteps(CONTROL_SEQUENCE_STEPS);
    optimizer.setCostThreshold(COST_THRESHOLD);
    optimizer.setGoalObjectName(GOAL_OBJECT_NAME);
    optimizer.setMaxOptimizationTime(TOTAL_TIME_LIMIT);
    optimizer.setTimeOutForHelpLimit(TIMEOUT_FOR_HELP);

    optimizer.setAdaptive(ADAPTIVE);
    optimizer.setAdaptiveLocalMinima(ADAPTIVE_LOCAL_MINIMA);
    optimizer.enablePredictiveTimeout(PREDICTIVE);
    optimizer.setThresholdPrediction(THRESHOLD_PREDICTION);
    optimizer.setAdaptiveNumberOfConesecutiveNonDecreases(NUMBER_OF_CONSECUTIVE_NON_DECREASES);

    State initialStartState = globalMujocoHelper->getState();
    optimizer.setInitialState(initialStartState);
    optimizer.logIndividualCosts();
}

void solve(bool saveResults) {
    ros::NodeHandle n;

    ros::ServiceClient releaseHuman = n.serviceClient<pgf::HumanRelease>("release");
    pgf::HumanRelease signal;
    signal.request.name = ROBOT_NAME;

    StraightLineInteractiveOptimizer optimizer(globalMujocoHelper, PLANE_HIGH_X, PLANE_LOW_X, PLANE_HIGH_Y, PLANE_LOW_Y);
    setupOptimizer(optimizer);
    optimizer.setRobotName(ROBOT_NAME);

    Result result = NullResult(false);

    bool humanAssigned = false;
    int experimentId = getNextExperimentId();
    int actionId = getNextActionId() - 1;
    string actionType = "reach";
    double totalTime = 0.0;
    double totalInteractionTime = 0.0;

    while (!optimizer.reachedPrimaryGoal()) {
        double totalRunningTime = totalTime + totalInteractionTime;
        if(totalRunningTime >= TOTAL_TIME_LIMIT) {
            printf("Timed-out, quiting.\n");
            break;
        }

        globalMujocoHelper->resetSimulation();
        double remainingPlanningTime = TOTAL_TIME_LIMIT - totalRunningTime;
        optimizer.setMaxOptimizationTime(remainingPlanningTime);
        optimizer.logIndividualCosts();

        glfwSetWindowTitle(window, "OPTIMIZATION IN PROGRESS");
        result = optimizer.optimize();
        

        glfwSetWindowTitle(window, "OPTIMIZATION DONE");
        totalTime += result.getOptimizationTime();

        if(saveResults) {
            actionId++;
            string outcome = result.isSuccessful() ? "success" : "failure";
            writeAction(experimentId, actionId, actionType, outcome, result.getOptimizationTime(), INTERACTION_TIME);

            vector<double> costSequence = result.getCostSequence();
            for(double cost : costSequence) {
                writeCostSequence(actionId, cost);
            }
        }

        if (result.isSuccessful()) {
            globalMujocoHelper->resetSimulation();
            auto controlSequence = result.getControlSequence();

            finishedExecution = false;
            glfwSetWindowTitle(window, "EXECUTING SOLUTION");
            thread t3(executeSolutionRealTime, controlSequence, optimizer.getActionDuration());

            while (!finishedExecution) {
                render();
            }

            t3.join();

            std::this_thread::sleep_for(std::chrono::seconds(1));

            globalMujocoHelper->saveState();
            State startState = globalMujocoHelper->getState();
            optimizer.setInitialState(startState);
            optimizer.updateMujocoHelpers(globalMujocoHelper);

            if (!optimizer.reachedPrimaryGoal()) {
                finishedExecution = false;
                thread t4(moveBack);

                while (!finishedExecution) {
                    render();
                }

                t4.join();

                // Changed to system happened since last time.
                globalMujocoHelper->saveState();
                State newStartState = globalMujocoHelper->getState();
                optimizer.setInitialState(newStartState);
                optimizer.updateMujocoHelpers(globalMujocoHelper);
                optimizer.setGoalObjectName(GOAL_OBJECT_NAME);
                actionType = "reach";
                INTERACTION_TIME = 0.0; // Here we are interested in per action interaction time, not overall interaction time.
            }
        } else {
            optimizer.setGoalObjectName(GOAL_OBJECT_NAME);
            actionType = "reach";
        }

        if(totalTime + totalInteractionTime >= TOTAL_TIME_LIMIT || optimizer.reachedPrimaryGoal()) {
            if(optimizer.reachedPrimaryGoal())
                printf("Success!\n");
            else
                printf("Timed-out, quiting.\n");
            break;
        }

        humanAssigned = optimizer.isHumanAssigned();

        // Request human-input.
        if (!optimizer.isOptimizing() && !result.isSuccessful() && humanAssigned) {
            glfwSetWindowPos(window, 550, 100);
            glfwShowWindow(window);
            glfwSetWindowFocusCallback(window, window_focus_callback);
            glfwSetWindowTitle(window, ("ROBOT: " + ROBOT_NAME + "- OPERATOR: SELECT AN OBJECT TO PUSH").c_str());
            INTERACTION_TIME = 0.0; // Here we are interested in per action interaction time, not overall interaction time.

            plannerIsIdle = true;
            plannerWaitingStartClock = std::chrono::high_resolution_clock::now();

            string objectNameSelected = getObjectSelection(MOVABLE_OBSTACLE_NAMES);

            if (objectNameSelected == GOAL_OBJECT_NAME) {
                State startState = globalMujocoHelper->getState();
                optimizer.setInitialState(startState);
                optimizer.setGoalObjectName(GOAL_OBJECT_NAME);
                actionType = "reach";
            } else {
                glfwSetWindowTitle(window, "OPERATOR: SELECT A PUSHING POINT");
                auto clickPosition = getPushPosition();
                double positionToPushObjectX = get<0>(clickPosition);
                double positionToPushObjectY = get<1>(clickPosition);

                // Make this a sub-goal.
                optimizer.setGoalObjectName(objectNameSelected);
                optimizer.setSubGoal(positionToPushObjectX, positionToPushObjectY);
                actionType = "push";
            }

            glfwSetWindowTitle(window, "OPERATOR: INPUT RECORDED... OPTIMIZING");

            auto plannerWaitingFinishClock = std::chrono::high_resolution_clock::now();
            plannerIsIdle = false;

            // There was idle time
            if (humanReturnedStartClock && plannerWaitingStartClock < humanReturnedStartClock) {
                INTERACTION_TIME += formatDurationToSeconds(humanReturnedStartClock.get(), plannerWaitingFinishClock);
                totalInteractionTime += INTERACTION_TIME;
            } else {
                INTERACTION_TIME += formatDurationToSeconds(plannerWaitingStartClock, plannerWaitingFinishClock);
                totalInteractionTime += INTERACTION_TIME;
            }

            optimizer.unAssignHuman();
            if (releaseHuman.call(signal)) { // Send signal that human is now free.
                cout << "Human released" << endl;
                humanAssigned = false;
                glfwHideWindow(window);
            } else {
                ROS_ERROR("Failed to release human");
            }
        }
    }

    glfwSetWindowTitle(window, "FINISHED");

    if(humanAssigned) {
        if (releaseHuman.call(signal)) { // Send signal that human is now free.
            cout << "Human released" << endl;
            glfwHideWindow(window);
        } else {
            ROS_ERROR("Failed to release human");
        }
    }

    // Write the actual vs predicted values
    if(PREDICTIVE) {
        auto autonomousPredictions = optimizer.getAutonomousPredictions();
        auto humanPredictions = optimizer.getHumanPredictions();
        auto actualCosts = optimizer.getActualCosts();
        savePredictionData(experimentId, actualCosts, autonomousPredictions, humanPredictions);
    }

    // Write experiment result outcome.
    if(saveResults) {
        updateNextActionId(actionId);
        string outcome = optimizer.reachedPrimaryGoal() ? "success" : "failure";
        string updateType = optimizer.isProbabilisticUpdate() ? "probabilistic"  : "greedy";
        writeExperiment(experimentId, outcome, updateType, totalTime, optimizer.getTotalTimeWaitingForHuman(), optimizer.getTotalTimeWastedWaitingForHuman());
        updateNextExperimentId(experimentId);
    }
}

int main(int argc, char **argv) {
    isZoomLocked = true;

    if (argc < 2 || (strcmp(argv[1], "--help") == 0)) {
        printf("Usage:\n");
        printf("rosrun   project_name   program_name   model_name   number_of_movable_objects\n");
        return 1;
    }

    mj_activate(MJ_KEY_PATH.c_str());

    SCENE_NAME = argv[1];
    NUMBER_OF_MOVABLE_OBJECTS = atoi(argv[2]);

    bool save_results = true;
    int parallelWindowIndex = 1;
    ROBOT_NAME = argv[3];
    EXPERIMENT_PATH = PROJECT_ROOT_PATH + "experiments/results/" + ROBOT_NAME + "/";

	ros::init(argc, argv, "bidding_system" + ROBOT_NAME);

    MOVABLE_OBSTACLE_NAMES = getObjectNamesFromNumberOfObjects(NUMBER_OF_MOVABLE_OBJECTS);

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    initMujocoWithCustomWindowFrom(SCENE_PATH + SCENE_NAME, parallelWindowIndex);
    glfwHideWindow(window);

    globalMujocoHelper->enableGripperDOF();
    globalMujocoHelper->setMovableObjectNames(MOVABLE_OBSTACLE_NAMES);
    globalMujocoHelper->forward();
    globalMujocoHelper->saveState();

    solve(save_results);
}
