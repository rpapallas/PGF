#ifndef EXECFUNCS
#define EXECFUNCS

#include "../utils/MujocoGlobal.cpp"
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Twist.h>
#include "../src/discrete/MoveInFreeSpaceOptimizer.cpp"

tuple<ControlSequence, double> moveRobotBackWithTrajectoryOptimization(vector<string> movableObstacleNames,
                                                                       vector<string> staticObstacleNames,
                                                                       const double planeHighX,
                                                                       const double planeLowX,
                                                                       const double planeHighY,
                                                                       const double planeLowY,
                                                                       const double tableZ) {

    printf("Moving robot backwards, optimization starting...\n");

    // Find the opposite direction of the robot's end-effector.
    auto endEffectorPosition = globalMujocoHelper->getSitePosition(
            "ee_point_1"); // This is the position of the end-effector.
    double endEffectorX = endEffectorPosition[0];
    double endEffectorY = endEffectorPosition[1];

    auto site2 = globalMujocoHelper->getSitePosition("ee_point_2");
    double site2_x = site2[0];
    double site2_y = site2[1];

    // Direction vector
    double eeVectorX = site2_x - endEffectorX;
    double eeVectorY = site2_y - endEffectorY;
    vector<double> directionVector = {eeVectorX, eeVectorY};
    vector<double> unitDirectionVector = globalMujocoHelper->unitVectorOf(directionVector);
    vector<double> oppositeUnitDirectionVector = {-unitDirectionVector[0], -unitDirectionVector[1]};

    oppositeUnitDirectionVector[0] *= 0.07;
    oppositeUnitDirectionVector[1] *= 0.07;

    int TRAJECTORY_ROLLOUTS = 5;
    int MAX_ITERATIONS = 5;
    double COST_THRESHOLD = 10.0;
    int TRAJECTORY_DURATION = 1;
    int CONTROL_SEQUENCE_STEPS = 5;
    vector<double> VARIANCE_VECTOR = {0.03, 0.03, 0.02};

    MoveInFreeSpaceOptimizer optimizer(globalMujocoHelper, planeHighX, planeLowX, planeHighY, planeLowY, tableZ);
    optimizer.setNumberOfNoisyTrajectoryRollouts(TRAJECTORY_ROLLOUTS);
    optimizer.setMaxIterations(MAX_ITERATIONS);
    optimizer.setCostThreshold(COST_THRESHOLD);
    optimizer.setTrajectoryDuration(TRAJECTORY_DURATION);
    optimizer.setSamplingVarianceVector(&VARIANCE_VECTOR);
    optimizer.setStaticObstacleNames(&staticObstacleNames);
    optimizer.setMovableObstacleNames(&movableObstacleNames);
    optimizer.setControlSequenceSteps(CONTROL_SEQUENCE_STEPS);

    double initialX = globalMujocoHelper->getRobotXpos();
    double initialY = globalMujocoHelper->getRobotYpos();
    double desiredX = initialX + oppositeUnitDirectionVector[0];
    double desiredY = initialY + oppositeUnitDirectionVector[1];
    double desiredYaw = 0.0;

    optimizer.setPointToMoveTo(desiredX, desiredY, desiredYaw);
    State startState = globalMujocoHelper->getState();
    optimizer.setInitialState(startState);

    Result output = optimizer.optimize();
    bool isSuccessful = output.isSuccessful();

    ControlSequence controlSequence(CONTROL_SEQUENCE_STEPS);
    double duration = optimizer.getActionDuration();

    if (isSuccessful) {
        controlSequence = output.getControlSequence();
    } else {
        printf("Failed to move backwards.\n");
    }

    return make_tuple(controlSequence, duration);
}

void executeSolutionRealTime(ControlSequence controlSequence, double propagationStepSize) {
    for (Control currentControl : controlSequence.getControls()) {
        vector<double> initialGripperJointValues = globalMujocoHelper->getCurrentJointValuesForGripper();
        vector<double> finalGripperJointValues = globalMujocoHelper->getJointValuesForGripper(
                currentControl.getGripperDOF()); // Fully-closed

        double steps = propagationStepSize / globalMujocoHelper->getTimeStep();

        vector<double> diffGripperJointValues = finalGripperJointValues;
        for (unsigned int i = 0; i < initialGripperJointValues.size(); ++i) {
            diffGripperJointValues[i] -= initialGripperJointValues[i];
        }

        for (int step = 0; step < steps; ++step) {
            globalMujocoHelper->setRobotVelocity(currentControl.getLinearX(), currentControl.getLinearY(),
                                                 currentControl.getAngularZ());

            // Set gripper DOF
            vector<double> stepGripperJointValues = initialGripperJointValues;
            for (unsigned int i = 0; i < stepGripperJointValues.size(); ++i) {
                stepGripperJointValues[i] += diffGripperJointValues[i] * (step / steps);
            }

            globalMujocoHelper->setGripperJointValues(stepGripperJointValues);

            globalMujocoHelper->step();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    finishedExecution = true;
}

#endif
