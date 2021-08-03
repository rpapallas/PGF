#ifndef PUSHTRAJOPT
#define PUSHTRAJOPT

#include "../TrajectoryOptimiserBase.cpp"

class PushOptimizer : public OptimizerBase {
public:
    PushOptimizer(shared_ptr<MujocoHelper> mujocoHelper,
                  double plane_high_x,
                  double plane_low_x,
                  double plane_high_y,
                  double plane_low_y,
                  double table_z) :
            OptimizerBase(mujocoHelper),
            _PLANE_HIGH_X(plane_high_x),
            _PLANE_LOW_X(plane_low_x),
            _PLANE_HIGH_Y(plane_high_y),
            _PLANE_LOW_Y(plane_low_y),
            _TABLE_Z(table_z) {
        _USE_PROBABILISTIC_UPDATE = false;
    }

    void setGoalObjectName(string name) {
        _goalObjectName = name;
    }

    void setGoalRegion(double x, double y, double radius) {
        _goalRegionCentroidX = x;
        _goalRegionCentroidY = y;
        _goalRegionRadius = radius;
    }

private:
    bool isGoalAchieved(State state) {
        return checkIfObjectIsNearTheGoalRegion(state);
    }

    double getDistanceOfObjectToRegion(shared_ptr<MujocoHelper> mujocoHelper) {
        double goalObjectX = mujocoHelper->getBodyXpos(_goalObjectName);
        double goalObjectY = mujocoHelper->getBodyYpos(_goalObjectName);

        double deltaX = goalObjectX - _goalRegionCentroidX;
        double deltaY = goalObjectY - _goalRegionCentroidY;

        return sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    double getDistanceOfObjectToRegion() {
        return getDistanceOfObjectToRegion(_mujocoHelper);
    }

    bool checkIfObjectIsNearTheGoalRegion(State state) {
        setMuJoCoTo(state);
        double distance = getDistanceOfObjectToRegion();
        resetToInitialState();
        return distance < _goalRegionRadius;
    }

    CostSequence cost(StateSequence stateSequence, ControlSequence controlSequence) {
        return cost(stateSequence, controlSequence, _mujocoHelper);
    }

    CostSequence cost(StateSequence stateSequence, ControlSequence /* controlSequence */, shared_ptr<MujocoHelper> mujocoHelper) {
        if (!_movableObstacleNames)
            throw std::invalid_argument("You should set a list of movable obstacle names!");

        CostSequence costSequence(_n); // We don't have a cost for the first step.
        for (unsigned int i = 0; i < stateSequence.size(); ++i) {
            if (i == stateSequence.size() - 1) { // Final state
                auto currentState = stateSequence.getState(i);
                double cost1 = distanceToGoalCost(currentState, mujocoHelper);
                double cost2 = intermediateCosts(currentState, mujocoHelper);
                costSequence.addCost(cost1 + cost2);
            } else {
                auto currentState = stateSequence.getState(i);
                double cost1 = intermediateCosts(currentState, mujocoHelper);
                costSequence.addCost(cost1);
            }
        }

        return costSequence;
    }

    double intermediateCosts(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        double edge = edgeCost(currentState, mujocoHelper);
        double highForce = highForceToStaticObstacleCost(currentState, mujocoHelper);

        double collision = 0.0;
        if (_staticObstacleNames)
            collision = collisionCost(currentState, mujocoHelper);

        return edge + collision + highForce;
    }

    double distanceToGoalCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        double distanceOfGoalObjectToGoalRegion = getDistanceOfObjectToRegion(mujocoHelper);

        if (distanceOfGoalObjectToGoalRegion < _goalRegionRadius) {
            return 0.0;
        }

        double costValue = _REACHED_GOAL_WEIGHT * distanceOfGoalObjectToGoalRegion;

        return costValue;
    }

    float edgeCost(State state, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(state, mujocoHelper);

        for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
            double objectXpos = mujocoHelper->getBodyXpos(_movableObstacleNames->at(i));
            double objectYpos = mujocoHelper->getBodyYpos(_movableObstacleNames->at(i));

            if (objectXpos > _PLANE_HIGH_X || objectXpos < _PLANE_LOW_X)
                return _EDGE_COST;

            if (objectYpos > _PLANE_HIGH_Y || objectYpos < _PLANE_LOW_Y)
                return _EDGE_COST;
        }

        return 0.0;
    }

    float highForceToStaticObstacleCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);

        if (mujocoHelper->isHighForceAppliedToShelf()) {
            return _HIGH_FORCE_COST;
        }

        return 0.0;
    }

    float disturbanceCost(State currentState, State nextState, shared_ptr<MujocoHelper> mujocoHelper) {
        double costValue = 0.0;

        setMuJoCoTo(nextState, mujocoHelper);
        vector<tuple<double, double>> postPositions;
        for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
            double objectXpos = mujocoHelper->getBodyXpos(_movableObstacleNames->at(i));
            double objectYpos = mujocoHelper->getBodyYpos(_movableObstacleNames->at(i));
            postPositions.push_back(make_tuple(objectXpos, objectYpos));
        }

        setMuJoCoTo(currentState, mujocoHelper);
        int index = 0;

        for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
            double preObjectXpos = mujocoHelper->getBodyXpos(_movableObstacleNames->at(i));
            double preObjectYpos = mujocoHelper->getBodyYpos(_movableObstacleNames->at(i));

            double postObjectXpos = get<0>(postPositions[index]);
            double postObjectYpos = get<1>(postPositions[index]);

            double deltaX = postObjectXpos - preObjectXpos;
            double deltaY = postObjectYpos - preObjectYpos;
            costValue += sqrt(deltaX * deltaX + deltaY * deltaY);
            index++;
        }

        return _DISTURBANCE_WEIGHT * (costValue * costValue);
    }

    float collisionCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        for (unsigned int i = 0; i < _staticObstacleNames->size(); ++i) {
            if (mujocoHelper->isRobotInContact(_staticObstacleNames->at(i))) {
                return _COLLISION_COST;
            }
        }

        return 0.0;
    }

    ControlSequence createCandidateControlSequence() {
        double goalObjectX = _goalRegionCentroidX;
        double goalObjectY = _goalRegionCentroidY;

        auto endEffectorPosition = _mujocoHelper->getSitePosition(
                "ee_point_1"); // This is the position of the end-effector.
        double endEffectorX = endEffectorPosition[0];
        double endEffectorY = endEffectorPosition[1];

        auto site2 = _mujocoHelper->getSitePosition("ee_point_2");
        double site2_x = site2[0];
        double site2_y = site2[1];

        // Direction vector
        double eeVectorX = site2_x - endEffectorX;
        double eeVectorY = site2_y - endEffectorY;
        vector<double> directionVector = {eeVectorX, eeVectorY};
        vector<double> unitDirectionVector = _mujocoHelper->unitVectorOf(directionVector);

        // Find the vector of the end-effector to the goal object.
        double eeToGoalX = goalObjectX - endEffectorX;
        double eeToGoalY = goalObjectY - endEffectorY;
        vector<double> eeToGoalVector = {eeToGoalX, eeToGoalY};
        vector<double> unitEeToGoalVector = _mujocoHelper->unitVectorOf(eeToGoalVector);

        // Calculate the angle between them.
        double angle = acos(_mujocoHelper->dotProduct(unitDirectionVector, unitEeToGoalVector)) * 0.08;

        unitEeToGoalVector[0] *= 0.03;
        unitEeToGoalVector[1] *= 0.03;

        double x = eeToGoalX - unitEeToGoalVector[0];
        double y = eeToGoalY - unitEeToGoalVector[1];

        double linearX = x / _trajectoryDuration;
        double linearY = y / _trajectoryDuration;
        double angularZ = angle / _trajectoryDuration;

        // At this stage where we push the object, we could initially
        // approach the object using open-finger such that the object falls
        // inside the fingers and therefore during this pushing we should
        // keep the hand open, if however we approached the object using
        // closed-fingers (pushing sideways) then we should keep the
        // fingers closed during this push. Either way, the current gripper
        // DOF Value (getGripperDOFValue) will give us the appropriate result.
        double gripperDOFValue = _initialState.getGripperDOFValue();

        Control velocities(linearX, linearY, angularZ, gripperDOFValue);
        ControlSequence controlSequence = ControlSequence(_n);
        for (int i = 0; i < _n; ++i) {
            controlSequence.addControl(velocities);
        }

        return controlSequence;
    }

    const double _EDGE_COST = 500;
    const double _ACCELERATION_WEIGHT = 0.000001;
    const double _DISTURBANCE_WEIGHT = 0.000001;
    const double _COLLISION_COST = 500; // Any static obstacle collision.
    const double _REACHED_GOAL_WEIGHT = 5000;
    const double _HIGH_FORCE_COST = 500;

    // Plane limits
    const double _PLANE_HIGH_X;
    const double _PLANE_LOW_X;
    const double _PLANE_HIGH_Y;
    const double _PLANE_LOW_Y;
    const double _TABLE_Z;

    string _goalObjectName = "object_3";
    double _goalRegionCentroidX = 0.0;
    double _goalRegionCentroidY = 0.0;
    double _goalRegionRadius = 0.0;
};

#endif

#ifndef APPROACHTRAJOPT
#define APPROACHTRAJOPT

#include "../TrajectoryOptimiserBase.cpp"
//TODO: Move this class to Utils
#include "../../../sampling_based_planning/src/ApproachingRobotState.cpp"

class ApproachOptimizer : public OptimizerBase {
public:
    ApproachOptimizer(shared_ptr<MujocoHelper> mujocoHelper,
                      double plane_high_x,
                      double plane_low_x,
                      double plane_high_y,
                      double plane_low_y,
                      double table_z) :
            OptimizerBase(mujocoHelper),
            _PLANE_HIGH_X(plane_high_x),
            _PLANE_LOW_X(plane_low_x),
            _PLANE_HIGH_Y(plane_high_y),
            _PLANE_LOW_Y(plane_low_y),
            _TABLE_Z(table_z) {
        _USE_PROBABILISTIC_UPDATE = true;
    }

    void setGoalObjectName(string name) {
        _goalObjectName = name;
    }

    void setGoalRegion(double x, double y, double radius) {
        _goalRegionCentroidX = x;
        _goalRegionCentroidY = y;
        _goalRegionRadius = radius;
    }

private:
    bool isGoalAchieved(State state) {
        return checkIfRobotIsNearTheGoalState(state);
    }

    bool checkIfRobotIsNearTheGoalState(State state) {
        setMuJoCoTo(state);
        double distance = getDistanceToGoalState(_mujocoHelper);
        resetToInitialState();
        return distance < 0.05;
    }

    CostSequence cost(StateSequence stateSequence, ControlSequence controlSequence) {
        return cost(stateSequence, controlSequence, _mujocoHelper);
    }

    CostSequence cost(StateSequence stateSequence, ControlSequence /* controlSequence */, shared_ptr<MujocoHelper> mujocoHelper) {
        if (!_movableObstacleNames)
            throw std::invalid_argument("You should set a list of movable obstacle names!");

        CostSequence costSequence(_n);
        for (unsigned int i = 0; i < stateSequence.size(); ++i) {
            if (i == stateSequence.size() - 1) { // Final state
                auto currentState = stateSequence.getState(i);
                double cost1 = distanceToGoalCost(currentState, mujocoHelper);
                double cost2 = intermediateCosts(currentState, mujocoHelper);
                costSequence.addCost(cost1 + cost2);
            } else {
                auto currentState = stateSequence.getState(i);
                double cost1 = intermediateCosts(currentState, mujocoHelper);
                costSequence.addCost(cost1);
            }
        }

        return costSequence;
    }

    double intermediateCosts(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        double edge = edgeCost(currentState, mujocoHelper);
        double highForce = highForceToStaticObstacleCost(currentState, mujocoHelper);

        double collision = 0.0;
        if (_staticObstacleNames)
            collision = collisionCost(currentState, mujocoHelper);

        return edge + collision + highForce;
    }

    double getDistanceToGoalState(shared_ptr<MujocoHelper> mujocoHelper) {
        double robotCurrentX = mujocoHelper->getRobotXpos();
        double robotCurrentY = mujocoHelper->getRobotYpos();

        double deltaX = _approachingX - robotCurrentX;
        double deltaY = _approachingY - robotCurrentY;

        double dist = sqrt(deltaX * deltaX + deltaY * deltaY);
        return dist;
    }

    double distanceToGoalCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        double distanceToGoalRobotState = getDistanceToGoalState(mujocoHelper);

        if (distanceToGoalRobotState < 0.05) {
            return 0.0;
        }

        double costValue = _REACHED_GOAL_WEIGHT * distanceToGoalRobotState;

        return costValue;
    }


    float edgeCost(State state, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(state, mujocoHelper);

        for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
            double objectXpos = mujocoHelper->getBodyXpos(_movableObstacleNames->at(i));
            double objectYpos = mujocoHelper->getBodyYpos(_movableObstacleNames->at(i));
            double objectZpos = mujocoHelper->getBodyZpos(_movableObstacleNames->at(i));

            if (objectZpos < _TABLE_Z)
                return _EDGE_COST;

            if (objectXpos > _PLANE_HIGH_X || objectXpos < _PLANE_LOW_X)
                return _EDGE_COST;

            if (objectYpos > _PLANE_HIGH_Y || objectYpos < _PLANE_LOW_Y)
                return _EDGE_COST;
        }

        return 0.0;
    }

    float highForceToStaticObstacleCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);

        if (mujocoHelper->isHighForceAppliedToShelf()) {
            return _HIGH_FORCE_COST;
        }

        return 0.0;
    }

    float disturbanceCost(State currentState, State nextState, shared_ptr<MujocoHelper> mujocoHelper) {
        double costValue = 0.0;

        setMuJoCoTo(nextState, mujocoHelper);
        vector<tuple<double, double>> postPositions;
        for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
            double objectXpos = mujocoHelper->getBodyXpos(_movableObstacleNames->at(i));
            double objectYpos = mujocoHelper->getBodyYpos(_movableObstacleNames->at(i));
            postPositions.push_back(make_tuple(objectXpos, objectYpos));
        }

        setMuJoCoTo(currentState, mujocoHelper);
        int index = 0;

        for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
            double preObjectXpos = mujocoHelper->getBodyXpos(_movableObstacleNames->at(i));
            double preObjectYpos = mujocoHelper->getBodyYpos(_movableObstacleNames->at(i));

            double postObjectXpos = get<0>(postPositions[index]);
            double postObjectYpos = get<1>(postPositions[index]);

            double deltaX = postObjectXpos - preObjectXpos;
            double deltaY = postObjectYpos - preObjectYpos;
            costValue += sqrt(deltaX * deltaX + deltaY * deltaY);
            index++;
        }

        return _DISTURBANCE_WEIGHT * (costValue * costValue);
    }

    float collisionCost(State currentState, shared_ptr<MujocoHelper> mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        for (unsigned int i = 0; i < _staticObstacleNames->size(); ++i) {
            if (mujocoHelper->isRobotInContact(_staticObstacleNames->at(i))) {
                return _COLLISION_COST;
            }
        }

        return 0.0;
    }


    vector<ApproachingRobotState>
    getApproachingPositions() {
        mjtGeom geomType = _mujocoHelper->getGeomTypeFromBodyName(_goalObjectName);

        if (geomType == mjGEOM_CYLINDER) {
            return getApproachingPositionsForCylinder();
        } else if (geomType == mjGEOM_BOX) {
            return getApproachingPositionsForBox();
        } else {
            throw std::logic_error("Approaching states can only be calculated for boxes and cylinders.");
        }
    }

    Eigen::Vector3d getEndEffectorTransformToObject(double goal_x,
                                                    double goal_y,
                                                    double object_x,
                                                    double object_y,
                                                    double scale) {
        double distance_x = object_x - goal_x;
        double distance_y = object_y - goal_y;
        double distance_z = 0.0;

        double magnitude = sqrt(distance_x * distance_x + distance_y * distance_y + distance_z * distance_z);
        double unit_vector[3] = {distance_x / magnitude, distance_y / magnitude, distance_z / magnitude};

        unit_vector[0] *= scale;
        unit_vector[1] *= scale;

        double x = object_x + unit_vector[0];
        double y = object_y + unit_vector[1];
        double z = 0.0;

        Eigen::Vector3d endEffectorGoalPosition(x, y, z);
        return endEffectorGoalPosition;
    }

    Eigen::MatrixXf getEndEffectorTransformAlignedToGoal(double object_x,
                                                         double object_y,
                                                         double goal_x,
                                                         double goal_y,
                                                         double scale) {
        Eigen::Vector3d target_position = getEndEffectorTransformToObject(goal_x, goal_y, object_x, object_y, scale);
        double target_x = target_position[0];
        double target_y = target_position[1];

        double dx = object_x - target_x;
        double dy = object_y - target_y;
        double dz = 0.0;

        double magnitude = sqrt(dx * dx + dy * dy + dz * dz);
        double unit_vector[3] = {dx / magnitude, dy / magnitude, dz / magnitude};

        Eigen::Vector3d x_rotation(unit_vector[0], unit_vector[1], unit_vector[2]);
        Eigen::Vector3d z_rotation(0, 0, 1);
        Eigen::Vector3d y_rotation2 = z_rotation.cross(x_rotation);

        double mag = sqrt(
                y_rotation2[0] * y_rotation2[0] + y_rotation2[1] * y_rotation2[1] + y_rotation2[2] * y_rotation2[2]);
        Eigen::Vector3d y_rotation = Eigen::Vector3d(y_rotation2[0] / mag, y_rotation2[1] / mag,
                                                     y_rotation2[2] / mag); // unit vector.

        // Orientation X
        Eigen::MatrixXf endEffectorInWorld = Eigen::MatrixXf::Identity(4, 4);
        endEffectorInWorld(0, 0) = x_rotation[0];
        endEffectorInWorld(1, 0) = x_rotation[1];
        endEffectorInWorld(2, 0) = x_rotation[2];

        // Orientation Y
        endEffectorInWorld(0, 1) = y_rotation[0];
        endEffectorInWorld(1, 1) = y_rotation[1];
        endEffectorInWorld(2, 1) = y_rotation[2];

        // Orientation Z
        endEffectorInWorld(0, 2) = z_rotation[0];
        endEffectorInWorld(1, 2) = z_rotation[1];
        endEffectorInWorld(2, 2) = z_rotation[2];

        // Position
        endEffectorInWorld(0, 3) = target_x;
        endEffectorInWorld(1, 3) = target_y;

        return endEffectorInWorld;
    }

    vector<ApproachingRobotState>
    getApproachingPositionsForCylinder() {
        double object_x = _mujocoHelper->getBodyXpos(_goalObjectName);
        double object_y = _mujocoHelper->getBodyYpos(_goalObjectName);
        double goal_x = _goalRegionCentroidX;
        double goal_y = _goalRegionCentroidY;

        // This is a transform that has the object in the gripper and the gripper's
        // orientation is such that is facing against the pushing goal position.
        // Think of it that the object is in the gripper's hand and if the robot
        // moves forward the direction is towards the goal position.
        Eigen::MatrixXf original1 = getEndEffectorTransformAlignedToGoal(object_x, object_y, goal_x, goal_y, 0.07);

        // Now find what is the correction rotation direction (+ or - z).
        Eigen::MatrixXf robotOriginalTransform = _mujocoHelper->getEndEffectorTransform();
        Eigen::MatrixXf matrixOffset = original1.inverse() * robotOriginalTransform;

        Eigen::Matrix3d m = Eigen::Matrix3d::Identity(3, 3);
        m(0, 0) = matrixOffset(0, 0);
        m(0, 1) = matrixOffset(0, 1);
        m(0, 2) = matrixOffset(0, 2);
        m(1, 0) = matrixOffset(1, 0);
        m(1, 1) = matrixOffset(1, 1);
        m(1, 2) = matrixOffset(1, 2);
        m(2, 0) = matrixOffset(2, 0);
        m(2, 1) = matrixOffset(2, 1);
        m(2, 2) = matrixOffset(2, 2);

        Eigen::AngleAxisd newAngleAxis(m);
        double angle = newAngleAxis.angle();

        // If the returned angle is > PI then the shortest angle is negative.
        int sign = 0;
        if (angle > M_PI || angle > -M_PI) {
            sign = -1;
        }

        if (angle < -M_PI || angle < M_PI) {
            sign = 1;
        }
        sign = sign * newAngleAxis.axis()[2];

        // Rotate +/- 90 degrees from original1 (in case where the original1
        // is facing the goal object from one of the shelf's sides, if such
        // case occurs we should rotate +/- 90 degrees torwards the original
        // orientation of the robot (assuming that the robot's intiial orientation
        // is facing the table's from the correct side (from the front). This
        // ensures that if the original1 transform is actually from the
        // table side (where obviously is the shelf side) we rotate +/- 90
        // degrees torwards the original orientation of the robot.
        Eigen::MatrixXf original2 = getRotatedRelativeTransform(original1, _goalObjectName, sign * M_PI / 2);

        // Now find transforms for pushing sideways
        Eigen::MatrixXf original3 = getEndEffectorTransformAlignedToGoal(object_x, object_y, goal_x, goal_y, 0.07);

        angle = sign * M_PI / 2;
        Eigen::MatrixXf rotationMatrix = Eigen::MatrixXf::Identity(4, 4);
        rotationMatrix(0, 0) = cos(angle);
        rotationMatrix(0, 1) = -sin(angle);
        rotationMatrix(1, 0) = sin(angle);
        rotationMatrix(1, 1) = cos(angle);
        Eigen::MatrixXf original4 = original3 * rotationMatrix;

        //transforms.push_back(original2);
        //transforms.push_back(original4);
        int type1 = 1; // This type has the object in hand.
        int type2 = 2; // This type has the object on the side of the hand.

        std::vector<std::tuple<double, double, double>> robotApproachingPositions;
        vector<ApproachingRobotState> robotApproachingStates;

        bool type1StateFound = false;

        if (!type1StateFound && !_mujocoHelper->checkRobotTransformForCollisions(original1, "shelf")) {
            ApproachingRobotState state(_mujocoHelper, original1, type1);
            if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                robotApproachingStates.push_back(state);
                type1StateFound = true;
            }
        }

        if (!type1StateFound) {
            // Get some more.
            Eigen::MatrixXf newTransform1 = original1;
            Eigen::MatrixXf newTransform2 = original1;

            // Check 50 degrees rotation +-
            for (int i = 0; i < 1; ++i) {
                double angleOfRotation1 = 0.174533; // 10 degrees
                double angleOfRotation2 = -0.174533; // 10 degrees
                newTransform1 = getRotatedRelativeTransform(newTransform1, _goalObjectName, angleOfRotation1);
                newTransform2 = getRotatedRelativeTransform(newTransform2, _goalObjectName, angleOfRotation2);

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform1, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform1, type1);
                    if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                        robotApproachingStates.push_back(state);
                        type1StateFound = true;
                        break;
                    }
                }

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform2, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform2, type1);
                    if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                        robotApproachingStates.push_back(state);
                        type1StateFound = true;
                        break;
                    }
                }
            }
        }

        bool ttt = false;
        if (!ttt && !_mujocoHelper->checkRobotTransformForCollisions(original2, "shelf")) {
            ApproachingRobotState state(_mujocoHelper, original2, type1);
            if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                robotApproachingStates.push_back(state);
                ttt = true;
            }
        }

        if (!ttt) {
            Eigen::MatrixXf newTransform1 = original2;
            Eigen::MatrixXf newTransform2 = original2;

            // Check 50 degrees rotation +-
            for (int i = 0; i < 1; ++i) {
                double angleOfRotation1 = 0.174533; // 10 degrees
                double angleOfRotation2 = -0.174533; // 10 degrees
                newTransform1 = getRotatedRelativeTransform(newTransform1, _goalObjectName, angleOfRotation1);
                newTransform2 = getRotatedRelativeTransform(newTransform2, _goalObjectName, angleOfRotation2);

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform1, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform1, type1);
                    if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                        robotApproachingStates.push_back(state);
                        ttt = true;
                        break;
                    }
                }

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform2, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform2, type1);
                    if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                        robotApproachingStates.push_back(state);
                        ttt = true;
                        break;
                    }
                }
            }
        }


        bool type2StateFound = false;

        if (!_mujocoHelper->checkRobotTransformForCollisions(original4, "shelf")) {
            ApproachingRobotState state(_mujocoHelper, original4, type2);
            if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                robotApproachingStates.push_back(state);
                type2StateFound = true;
            }
        }

        if (!type2StateFound) {
            Eigen::MatrixXf newTransform1 = original4;
            Eigen::MatrixXf newTransform2 = original4;

            // Check 50 degrees rotation +-
            for (int i = 0; i < 1; ++i) {
                double angleOfRotation1 = 0.174533; // 10 degrees
                double angleOfRotation2 = -0.174533; // 10 degrees
                newTransform1 = getRotatedRelativeTransform(newTransform1, _goalObjectName, angleOfRotation1);
                newTransform2 = getRotatedRelativeTransform(newTransform2, _goalObjectName, angleOfRotation2);

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform1, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform1, type2);
                    if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                        robotApproachingStates.push_back(state);
                        type2StateFound = true;
                        break;
                    }
                }

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform2, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform2, type2);
                    if (state.getYaw() < M_PI / 4 && state.getYaw() > -(M_PI / 4)) {
                        robotApproachingStates.push_back(state);
                        type2StateFound = true;
                        break;
                    }
                }
            }
        }

        return robotApproachingStates;
    }

    vector<ApproachingRobotState>
    getApproachingPositionsForBox() {
        double object_x = _mujocoHelper->getBodyXpos(_goalObjectName);
        double object_y = _mujocoHelper->getBodyYpos(_goalObjectName);
        double goal_x = _goalRegionCentroidX;
        double goal_y = _goalRegionCentroidY;

        // This is a transform that has the object in the gripper and the gripper's
        // orientation is such that is facing against the pushing goal position.
        // Think of it that the object is in the gripper's hand and if the robot
        // moves forward the direction is towards the goal position.
        Eigen::MatrixXf original1 = getEndEffectorTransformAlignedToGoal(object_x,
                                                                         object_y,
                                                                         goal_x,
                                                                         goal_y,
                                                                         0.15);

        // Now find what is the correction rotation direction (+ or - z).
        Eigen::MatrixXf robotOriginalTransform = _mujocoHelper->getEndEffectorTransform();
        Eigen::MatrixXf matrixOffset = original1.inverse() * robotOriginalTransform;

        Eigen::Matrix3d m = Eigen::Matrix3d::Identity(3, 3);
        m(0, 0) = matrixOffset(0, 0);
        m(0, 1) = matrixOffset(0, 1);
        m(0, 2) = matrixOffset(0, 2);
        m(1, 0) = matrixOffset(1, 0);
        m(1, 1) = matrixOffset(1, 1);
        m(1, 2) = matrixOffset(1, 2);
        m(2, 0) = matrixOffset(2, 0);
        m(2, 1) = matrixOffset(2, 1);
        m(2, 2) = matrixOffset(2, 2);

        Eigen::AngleAxisd newAngleAxis(m);
        double angle = newAngleAxis.angle();

        // If the returned angle is > PI then the shortest angle is negative.
        int sign = 0;
        if (angle > M_PI || angle > -M_PI) {
            sign = -1;
        }

        if (angle < -M_PI || angle < M_PI) {
            sign = 1;
        }
        sign = sign * newAngleAxis.axis()[2];

        // Now find transforms for pushing sideways
        Eigen::MatrixXf original3 = getEndEffectorTransformAlignedToGoal(object_x,
                                                                         object_y,
                                                                         goal_x,
                                                                         goal_y,
                                                                         0.15);

        angle = sign * M_PI / 2;
        Eigen::MatrixXf rotationMatrix = Eigen::MatrixXf::Identity(4, 4);
        rotationMatrix(0, 0) = cos(angle);
        rotationMatrix(0, 1) = -sin(angle);
        rotationMatrix(1, 0) = sin(angle);
        rotationMatrix(1, 1) = cos(angle);
        Eigen::MatrixXf original4 = original3 * rotationMatrix;

        int type1 = 1; // This type has the object in hand.
        int type2 = 2; // This type has the object on the side of the hand.

        std::vector<std::tuple<double, double, double>> robotApproachingPositions;
        vector<ApproachingRobotState> robotApproachingStates;

        bool type1StateFound = false;
        type1StateFound = true;

        if (!_mujocoHelper->checkRobotTransformForCollisions(original1, "shelf")) {
            ApproachingRobotState state(_mujocoHelper, original1, type1);
            robotApproachingStates.push_back(state);
            type1StateFound = true;
        }

        if (!type1StateFound) {
            // Get some more.
            Eigen::MatrixXf newTransform1 = original1;
            Eigen::MatrixXf newTransform2 = original1;

            // Check 60 degrees rotation +-
            for (int i = 0; i < 6; ++i) {
                double angleOfRotation1 = 0.174533; // 10 degrees
                double angleOfRotation2 = -0.174533; // 10 degrees
                newTransform1 = getRotatedRelativeTransform(newTransform1, _goalObjectName, angleOfRotation1);
                newTransform2 = getRotatedRelativeTransform(newTransform2, _goalObjectName, angleOfRotation2);

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform1, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform1, type1);
                    robotApproachingStates.push_back(state);
                    type1StateFound = true;
                    break;
                }

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform2, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform2, type1);
                    robotApproachingStates.push_back(state);
                    type1StateFound = true;
                    break;
                }
            }
        }

        bool type2StateFound = false;

        if (!_mujocoHelper->checkRobotTransformForCollisions(original4, "shelf")) {
            ApproachingRobotState state(_mujocoHelper, original4, type1);
            robotApproachingStates.push_back(state);
        }

        if (!type2StateFound) {
            Eigen::MatrixXf newTransform1 = original4;
            Eigen::MatrixXf newTransform2 = original4;

            // Check 60 degrees rotation +-
            for (int i = 0; i < 6; ++i) {
                double angleOfRotation1 = 0.174533; // 10 degrees
                double angleOfRotation2 = -0.174533; // 10 degrees
                newTransform1 = getRotatedRelativeTransform(newTransform1, _goalObjectName, angleOfRotation1);
                newTransform2 = getRotatedRelativeTransform(newTransform2, _goalObjectName, angleOfRotation2);

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform1, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform1, type2);
                    robotApproachingStates.push_back(state);
                    type2StateFound = true;
                    break;
                }

                if (!_mujocoHelper->checkRobotTransformForCollisions(newTransform2, "shelf")) {
                    ApproachingRobotState state(_mujocoHelper, newTransform2, type2);
                    robotApproachingStates.push_back(state);
                    type2StateFound = true;
                    break;
                }
            }
        }

        //auto filteredApproachingStates = filterOutSomeApproachingStates(robotApproachingStates);
        //return filteredApproachingStates;
        return robotApproachingStates;
    }

    Eigen::MatrixXf getRotatedRelativeTransform(Eigen::MatrixXf originalTransform, string objectName, double angle) {
        Eigen::MatrixXf rotationMatrix = Eigen::MatrixXf::Identity(4, 4);
        rotationMatrix(0, 0) = cos(angle);
        rotationMatrix(0, 1) = -sin(angle);
        rotationMatrix(1, 0) = sin(angle);
        rotationMatrix(1, 1) = cos(angle);

        Eigen::MatrixXf endEffectorInObject = _mujocoHelper->getBodyTransform(objectName).inverse() * originalTransform;
        Eigen::MatrixXf endEffectorInEndEffector = endEffectorInObject.inverse() * rotationMatrix * endEffectorInObject;
        Eigen::MatrixXf robotInWorld = _mujocoHelper->getBodyTransform(objectName) * endEffectorInObject;
        Eigen::MatrixXf finalTransform = robotInWorld * endEffectorInEndEffector;

        return finalTransform;
    }

    ControlSequence createCandidateControlSequence() {
        double robotInitialX = _mujocoHelper->getRobotXpos();
        double robotInitialY = _mujocoHelper->getRobotYpos();
        double robotInitialYaw = _mujocoHelper->getRobotYaw();

        double goalObjectInitialX = _mujocoHelper->getBodyXpos(_goalObjectName);
        double goalObjectInitialY = _mujocoHelper->getBodyYpos(_goalObjectName);

        auto robotApproachingPositions = getApproachingPositions();

        //double bestDistance = 10000000.0;
        double minCost = 10000000.0;
        ControlSequence bestControlSequence(_n);

        PushOptimizer pushOptimizer(_mujocoHelper, _PLANE_HIGH_X, _PLANE_LOW_X, _PLANE_HIGH_Y, _PLANE_LOW_Y, _TABLE_Z);
        pushOptimizer.setNumberOfNoisyTrajectoryRollouts(_numberOfNoisyTrajectoryRollouts);
        pushOptimizer.setMaxIterations(_maxIterations);
        pushOptimizer.setCostThreshold(_costThreshold);
        pushOptimizer.setTrajectoryDuration(_trajectoryDuration);
        pushOptimizer.setSamplingVarianceVector(_samplingVarianceVector);
        pushOptimizer.setStaticObstacleNames(_staticObstacleNames);
        pushOptimizer.setMovableObstacleNames(_movableObstacleNames);
        pushOptimizer.setControlSequenceSteps(_n);

        pushOptimizer.setGoalObjectName(_goalObjectName);
        pushOptimizer.setGoalRegion(_goalRegionCentroidX, _goalRegionCentroidY, _goalRegionRadius);

        for (unsigned int i = 0; i < robotApproachingPositions.size(); ++i) {
            double approachingX = robotApproachingPositions[i].getX();
            double approachingY = robotApproachingPositions[i].getY();
            double approachingYaw = robotApproachingPositions[i].getYaw();

            double deltaX = approachingX - robotInitialX;
            double deltaY = approachingY - robotInitialY;
            double deltaYaw = approachingYaw - robotInitialYaw;

            double linearX = deltaX / _trajectoryDuration;
            double linearY = deltaY / _trajectoryDuration;
            double angularZ = deltaYaw / _trajectoryDuration;


            // If the type of the approaching position is 1, then it means
            // that this approaching state is grasping the object in the
            // hand to push it, and therefore the gripper DOF should be
            // fully opened.
            ControlSequence controlSequence = ControlSequence(_n);
            if (robotApproachingPositions[i].getType() == 1) {
                Control velocitiesWithClosedGripper(linearX, linearY, angularZ, 255.0);
                Control velocitiesWithOpenGripper(linearX, linearY, angularZ, 0.0);

                for (int j = 0; j < _n - 2; ++j) {
                    controlSequence.addControl(velocitiesWithClosedGripper);
                }

                // During the very last few steps of the control sequence,
                // open the gripper as we would like the object to fall
                // inside our hand.
                for (int k = _n - 2; k < _n; ++k) {
                    controlSequence.addControl(velocitiesWithOpenGripper);
                }
            } else {
                Control velocities(linearX, linearY, angularZ, 255.0); // Gripper fully-closed.

                for (int k = 0; k < _n; ++k) {
                    controlSequence.addControl(velocities);
                }
            }

            State state = executeControlSequenceAndObtainState(controlSequence, _mujocoHelper);
            setMuJoCoTo(state);

            pushOptimizer.setInitialState(state);

            auto result = pushOptimizer.getInitialControlSequenceAndItsCost();
            double pushCost = get<2>(result).sum();

            auto approachRollout = trajectoryRollout(controlSequence);
            double approachCost = get<1>(approachRollout).sum();

            double goalObjectFinalX = _mujocoHelper->getBodyXpos(_goalObjectName);
            double goalObjectFinalY = _mujocoHelper->getBodyYpos(_goalObjectName);
            deltaX = goalObjectFinalX - goalObjectInitialX;
            deltaY = goalObjectFinalY - goalObjectInitialY;

            double goalObjectRearrangementDistance = sqrt(deltaX * deltaX + deltaY * deltaY);

            if (0.1 * pushCost + 0.3 * approachCost + 0.6 * goalObjectRearrangementDistance < minCost) {
                _approachingX = robotApproachingPositions[i].getX();
                _approachingY = robotApproachingPositions[i].getY();
                bestControlSequence = controlSequence;
                minCost = pushCost;
            }

            resetToInitialState();
        }

        return bestControlSequence;
    }

    const double _EDGE_COST = 500;
    const double _ACCELERATION_WEIGHT = 0.000001;
    const double _DISTURBANCE_WEIGHT = 0.000001;
    const double _COLLISION_COST = 500; // Any static obstacle collision.
    const double _REACHED_GOAL_WEIGHT = 5000;
    const double _HIGH_FORCE_COST = 500;

    // Plane limits
    const double _PLANE_HIGH_X;
    const double _PLANE_LOW_X;
    const double _PLANE_HIGH_Y;
    const double _PLANE_LOW_Y;
    const double _TABLE_Z;

    double _approachingX = 0.0;
    double _approachingY = 0.0;
    string _goalObjectName = "object_3";
    double _goalRegionCentroidX = 0.0;
    double _goalRegionCentroidY = 0.0;
    double _goalRegionRadius = 0.0;
};

#endif
