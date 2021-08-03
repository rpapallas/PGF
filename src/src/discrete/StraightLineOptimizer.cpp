#ifndef STRAIGHTTRAJOPT
#define STRAIGHTTRAJOPT

#include <utility>
#include "../TrajectoryOptimiserBase.cpp"
#include "HighLevelAction.cpp"

class StraightLineInteractiveOptimizer : public InteractiveOptimizerBase {
public:
    StraightLineInteractiveOptimizer(const shared_ptr<MujocoHelper> &mujocoHelper, double plane_high_x,
                                     double plane_low_x, double plane_high_y, double planer_low_y)
            : InteractiveOptimizerBase(mujocoHelper), _PLANE_HIGH_X(plane_high_x), _PLANE_LOW_X(plane_low_x),
              _PLANE_HIGH_Y(plane_high_y), _PLANE_LOW_Y(planer_low_y) {
        _USE_PROBABILISTIC_UPDATE = false;
    }

    void setGoalObjectName(string name) {
        _goalObjectName = std::move(name);
    }

protected:
    bool isPrimaryGoalAchieved(State state) override {
        return checkIfRobotGraspsGoalObject(state);
    }

    bool isSubGoalAchieved(State state) override {
        return checkIfObjectPushedToItsDesiredLocation(state);
    }

    bool isGoalAchieved(State state) override {
        return checkIfRobotGraspsGoalObject(state);
    }

    bool checkIfObjectPushedToItsDesiredLocation(State state) {
        setMuJoCoTo(state);
        double objectX = _mujocoHelper->getBodyXpos(_goalObjectName);
        double objectY = _mujocoHelper->getBodyYpos(_goalObjectName);

        double desiredX = _subGoalDesiredX;
        double desiredY = _subGoalDesiredY;

        double deltaX = desiredX - objectX;
        double deltaY = desiredY - objectY;

        double distance = sqrt(deltaX * deltaX + deltaY * deltaY);
        resetToInitialState();

        mjtGeom geomType = globalMujocoHelper->getGeomTypeFromBodyName(_goalObjectName);
        if (geomType == mjGEOM_BOX) {
            return distance < 0.05;
        } else if (geomType == mjGEOM_CYLINDER) {
            return distance < 0.1;
        }

        return distance < 0.1;
    }

    bool checkIfRobotGraspsGoalObject(State state) {
        setMuJoCoTo(state);
        double distance = _mujocoHelper->getDistanceOfEndEffectorTo(_goalObjectName);
        resetToInitialState();
        return distance < _DISTANCE_TO_GOAL_THRESHOLD;
    }

    ControlSequence createCandidateControlSequence() override {
        if (_subGoalSet) {
            return createCandidateControlSequenceForSubGoal();
        }

        return createCandidateControlSequenceForPrimaryGoal();
    }

    ControlSequence createCandidateControlSequenceForSubGoal() {
        // Approach
        // Goal Object for a sub-goal will be the pushing object (not the green object)
        auto goalObjectX = _mujocoHelper->getBodyXpos(_goalObjectName);
        auto goalObjectY = _mujocoHelper->getBodyYpos(_goalObjectName);

        // Direction vector
        double objectToPushPositionX = _subGoalDesiredX - goalObjectX;
        double objectToPushPositionY = _subGoalDesiredY - goalObjectY;
        vector<double> directionVector = {objectToPushPositionX, objectToPushPositionY};
        vector<double> unitDirectionVector = _mujocoHelper->unitVectorOf(directionVector);
        vector<double> oppositeUnitDirectionVector = {-unitDirectionVector[0], -unitDirectionVector[1]};


        mjtGeom geomType = globalMujocoHelper->getGeomTypeFromBodyName(_goalObjectName);
        if (geomType == mjGEOM_BOX) {
            oppositeUnitDirectionVector[0] *= 0.12;
            oppositeUnitDirectionVector[1] *= 0.12;
        } else if (geomType == mjGEOM_CYLINDER) {
            oppositeUnitDirectionVector[0] *= 0.07;
            oppositeUnitDirectionVector[1] *= 0.07;
        }

        // Since we are controlling the robot position and not the end-effector, we need to convert the end-effector
        // position to robot position.
        Eigen::MatrixXf desiredEndEffectorTransform = Eigen::MatrixXf::Identity(4, 4);
        desiredEndEffectorTransform(0, 3) = goalObjectX + oppositeUnitDirectionVector[0];
        desiredEndEffectorTransform(1, 3) = goalObjectY + oppositeUnitDirectionVector[1];

        Eigen::MatrixXf endEffectorInRobot = _mujocoHelper->getEndEffectorInRobotTransform();
        Eigen::MatrixXf robotInEndEffector = endEffectorInRobot.inverse();
        Eigen::MatrixXf desiredRobotTransform = desiredEndEffectorTransform * robotInEndEffector;

        double robotX = _mujocoHelper->getRobotXpos();
        double robotY = _mujocoHelper->getRobotYpos();
        double desiredRobotX = desiredRobotTransform(0, 3) - robotX;
        double desiredRobotY = desiredRobotTransform(1, 3) - robotY;
        double desiredYaw = 0.0;

        double linearX = desiredRobotX / (_trajectoryDuration / 2);
        double linearY = desiredRobotY / (_trajectoryDuration / 2);
        double angularZ = desiredYaw / (_trajectoryDuration / 2);

        Control velocitiesWithClosedGripper(linearX, linearY, angularZ, 255.0);
        ControlSequence controlSequence = ControlSequence(_n);
        for (int j = 0; j < _n / 2; ++j) {
            controlSequence.addControl(velocitiesWithClosedGripper);
        }


        // Push

        Eigen::MatrixXf desiredEndEffectorTransform2 = Eigen::MatrixXf::Identity(4, 4);
        desiredEndEffectorTransform2(0, 3) = _subGoalDesiredX;
        desiredEndEffectorTransform2(1, 3) = _subGoalDesiredY;
        Eigen::MatrixXf desiredRobotTransform2 = desiredEndEffectorTransform2 * robotInEndEffector;

        double endEffectorPushX = desiredRobotTransform2(0, 3) - desiredRobotTransform(0, 3);
        double endEffectorPushY = desiredRobotTransform2(1, 3) - desiredRobotTransform(1, 3);

        // Now scale pushing a bit because we want the object to be pushed to that region not the end-effector.
        vector<double> directionVector2 = {endEffectorPushX, endEffectorPushY};
        vector<double> unitDirectionVector2 = _mujocoHelper->unitVectorOf(directionVector2);
        unitDirectionVector2[0] *= 0.05;
        unitDirectionVector2[1] *= 0.05;
        endEffectorPushX = endEffectorPushX - unitDirectionVector2[0];
        endEffectorPushY = endEffectorPushY - unitDirectionVector2[1];

        linearX = endEffectorPushX / (_trajectoryDuration / 2);
        linearY = endEffectorPushY / (_trajectoryDuration / 2);
        angularZ = 0.0;

        Control velocitiesWithClosedGripper2(linearX, linearY, angularZ, 255.0);
        for (int j = 0; j < _n / 2; ++j) {
            controlSequence.addControl(velocitiesWithClosedGripper2);
        }

        return controlSequence;
    }

    ControlSequence createCandidateControlSequenceForPrimaryGoal() {
        // Current position of goal object.
        double goalObjectX = _mujocoHelper->getBodyXpos(_goalObjectName);
        double goalObjectY = _mujocoHelper->getBodyYpos(_goalObjectName);

        // Find the vector of the end-effector to the goal object.
        auto endEffectorPosition = _mujocoHelper->getSitePosition("ee_point_1");
        double endEffectorX = endEffectorPosition[0];
        double endEffectorY = endEffectorPosition[1];
        double eeToGoalX = goalObjectX - endEffectorX;
        double eeToGoalY = goalObjectY - endEffectorY;

        vector<double> goalVector = {eeToGoalX, eeToGoalY};
        vector<double> unitVector = _mujocoHelper->unitVectorOf(goalVector);
        unitVector[0] *= 0.04;
        unitVector[1] *= 0.04;
        eeToGoalX -= unitVector[0];
        eeToGoalY -= unitVector[1];

        // Now convert end-effector point into robot point.
        Eigen::MatrixXf desiredEndEffectorTransform = Eigen::MatrixXf::Identity(4, 4);
        desiredEndEffectorTransform(0, 3) = eeToGoalX;
        desiredEndEffectorTransform(1, 3) = eeToGoalY;
        Eigen::MatrixXf endEffectorInRobot = _mujocoHelper->getEndEffectorInRobotTransform();
        Eigen::MatrixXf robotInEndEffector = endEffectorInRobot.inverse();
        Eigen::MatrixXf desiredRobotTransform = desiredEndEffectorTransform * robotInEndEffector;

        // Calculate direction vector
        auto site2 = _mujocoHelper->getSitePosition("ee_point_0");
        double site2_x = site2[0];
        double site2_y = site2[1];
        double eeVectorX = endEffectorX - site2_x;
        double eeVectorY = endEffectorY - site2_y;
        vector<double> directionVector = {eeVectorX, eeVectorY};

        double x1 = directionVector[0];
        double y1 = directionVector[1];

        auto robotPosition = _mujocoHelper->getSitePosition("ee_point_0");
        double robotX = robotPosition[0];
        double robotY = robotPosition[1];
        double eeToGoalX2 = goalObjectX - robotX;
        double eeToGoalY2 = goalObjectY - robotY;
        vector<double> robotVector = {eeToGoalX2, eeToGoalY2};

        double x2 = robotVector[0];
        double y2 = robotVector[1];

        auto dot = x1 * x2 + y1 * y2;
        auto det = x1 * y2 - y1 * x2;
        double angle = 0.5 * atan2(det, dot);//  # atan2(y, x) or atan2(sin, cos)

        double linearX = eeToGoalX / _trajectoryDuration;
        double linearY = eeToGoalY / _trajectoryDuration;
        double angularZ = angle / _trajectoryDuration;

        Control velocitiesWithClosedGripper(linearX, linearY, angularZ, 255.0);
        Control velocitiesWithOpenGripper(linearX, linearY, angularZ, 0.0);

        int openGripperSteps;
        double distance = _mujocoHelper->getDistanceOfEndEffectorTo(_goalObjectName);
        if(distance < 0.15) // We don't want to close the gripper if already too close to goal.
            openGripperSteps = _n;
        else
            openGripperSteps = _n * 0.6;

        ControlSequence controlSequence = ControlSequence(_n);
        for (int j = 0; j < _n - openGripperSteps; ++j) {
            controlSequence.addControl(velocitiesWithClosedGripper);
        }

        // During the very last few steps of the control sequence,
        // open the gripper as we would like the object to fall
        // inside our hand.
        for (int k = _n - openGripperSteps; k < _n; ++k) {
            controlSequence.addControl(velocitiesWithOpenGripper);
        }

        return controlSequence;
    }

    CostSequence cost(StateSequence stateSequence, ControlSequence controlSequence) override {
        return cost(stateSequence, controlSequence, _mujocoHelper);
    }

    CostSequence
    cost(StateSequence stateSequence, const ControlSequence & /* controlSequence */,
         const shared_ptr<MujocoHelper> &mujocoHelper) {
        if (!_movableObstacleNames)
            throw invalid_argument("You should set a list of movable obstacle names!");

        CostSequence costSequence(_n); // We don't have a cost for the first step.
        for (unsigned int i = 0; i < stateSequence.size(); ++i) {
            auto currentState = stateSequence.getState(i);
            auto goalCost = 0.0;

            if (i == stateSequence.size() - 1) { // Final state
                double distanceToGoal = distanceToGoalCost(currentState, mujocoHelper);
                costSequence.addIndividualCost(distanceToGoal, 4);

                double extraGoal = extraGoalCost(currentState, mujocoHelper);
                costSequence.addIndividualCost(extraGoal, 5);

                goalCost = distanceToGoal + extraGoal;
            }

            double edge = edgeCost(currentState, mujocoHelper);
            costSequence.addIndividualCost(edge, 0);

            double highForce = highForceToObstaclesCost(currentState, mujocoHelper);
            costSequence.addIndividualCost(highForce, 1);

            double collision = 0.0;
            if (_staticObstacleNames)
                collision = collisionCost(currentState, mujocoHelper);
            costSequence.addIndividualCost(collision, 2);

            if(_print)
                printf("collision: %f, edge: %f, highForce: %f\n", collision, edge, highForce);

            double intermediateCost = edge + highForce + collision;


            double cost2 = extraIntermediateCost(currentState, mujocoHelper);
            costSequence.addIndividualCost(cost2, 3);

            costSequence.addCost(intermediateCost + cost2 + goalCost);
        }

        return costSequence;
    }

    double distanceToGoalCost(State currentState, const shared_ptr<MujocoHelper> &mujocoHelper) {
        if (_subGoalSet)
            return distanceToSubGoalCost(currentState, mujocoHelper);

        return distanceToPrimaryGoalCost(currentState, mujocoHelper);
    }

    double distanceToSubGoalCost(State currentState, const shared_ptr<MujocoHelper> &mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);

        double objectX = _mujocoHelper->getBodyXpos(_goalObjectName);
        double objectY = _mujocoHelper->getBodyYpos(_goalObjectName);

        double desiredX = _subGoalDesiredX;
        double desiredY = _subGoalDesiredY;

        double deltaX = desiredX - objectX;
        double deltaY = desiredY - objectY;

        double distance = sqrt(deltaX * deltaX + deltaY * deltaY);

        if (distance < 0.05) {
            return distance;
        }

        return _REACHED_GOAL_WEIGHT * distance;
    }

    double distanceToPrimaryGoalCost(State currentState, const shared_ptr<MujocoHelper> &mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        double distanceOfEndEffectorToGoalObject = mujocoHelper->getDistanceOfEndEffectorTo(_goalObjectName);
        resetToInitialState(mujocoHelper);

        if (distanceOfEndEffectorToGoalObject < _DISTANCE_TO_GOAL_THRESHOLD) {

            if(_print)
                printf("goal distance: %f\n", distanceOfEndEffectorToGoalObject);

            return distanceOfEndEffectorToGoalObject;
        }

        if(_print)
            printf("goal distance: %f with %f\n", _REACHED_GOAL_WEIGHT * distanceOfEndEffectorToGoalObject, distanceOfEndEffectorToGoalObject);

        return _REACHED_GOAL_WEIGHT * distanceOfEndEffectorToGoalObject;
    }

    float edgeCost(State state, const shared_ptr<MujocoHelper> &mujocoHelper) {
        setMuJoCoTo(state, mujocoHelper);

        double totalCost = 0.0;
        for (const auto &movableObstacleName : *_movableObstacleNames) {
            double objectXpos = mujocoHelper->getBodyXpos(movableObstacleName);
            double objectYpos = mujocoHelper->getBodyYpos(movableObstacleName);

            if (objectXpos > _PLANE_HIGH_X || objectXpos < _PLANE_LOW_X)
                totalCost += _EDGE_COST;

            if (objectYpos > _PLANE_HIGH_Y || objectYpos < _PLANE_LOW_Y)
                totalCost += _EDGE_COST;
        }

        resetToInitialState(mujocoHelper);
        return totalCost;
    }

    float highForceToObstaclesCost(State currentState, const shared_ptr<MujocoHelper> &mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);

        double forceCost = 0.0;

        for (const auto &objectName : *_movableObstacleNames) {
            vector<mjtNum *> bodyForceTorques = mujocoHelper->getBodyForceTorques(objectName);
            for (mjtNum *forceTorque : bodyForceTorques) {
                double forceMagnitude = sqrt(forceTorque[0] * forceTorque[0] + forceTorque[1] * forceTorque[1] + forceTorque[2] * forceTorque[2]);
                if(isinf(forceMagnitude)) {
                    forceCost += _HIGH_FORCE_COST * 100;
                }
                else if (forceMagnitude > 40.0) {
                    forceCost += _HIGH_FORCE_COST * forceMagnitude;
                }
            }
        }

        resetToInitialState(mujocoHelper);
        return forceCost;
    }

    float collisionCost(State currentState, const shared_ptr<MujocoHelper> &mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        for (const auto &staticObstacleName : *_staticObstacleNames) {
            if (mujocoHelper->isRobotInContact(staticObstacleName)) {
                resetToInitialState(mujocoHelper);
                return _COLLISION_COST;
            }
        }

        resetToInitialState(mujocoHelper);
        return 0.0;
    }

    virtual double extraIntermediateCost(State /*currentState*/, shared_ptr<MujocoHelper> /*mujocoHelper*/) {
        return 0.0;
    }

    virtual double extraGoalCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        double cost = 0.0;

        for(auto objectName : *_movableObstacleNames) {
            if(objectName != _goalObjectName) {
                double distance = mujocoHelper->getDistanceOfEndEffectorTo(objectName);
                if(distance < 0.075) {
                    cost +=  450 + distance;
                }
            }
        }

        resetToInitialState(mujocoHelper);
        return cost;
    }

    const double _DISTANCE_TO_GOAL_THRESHOLD = 0.06;
    const double _EDGE_COST = 300;
    const double _COLLISION_COST = 300; // Any static obstacle collision.
    const double _REACHED_GOAL_WEIGHT = 4000;
    const double _HIGH_FORCE_COST = 0.3;

    // Plane limits
    const double _PLANE_HIGH_X;
    const double _PLANE_LOW_X;
    const double _PLANE_HIGH_Y;
    const double _PLANE_LOW_Y;
};

#endif
