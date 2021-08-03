#ifndef FREESPACEOPT
#define FREESPACEOPT

#include "../TrajectoryOptimiserBase.cpp"

class MoveInFreeSpaceOptimizer : public OptimizerBase {
public:
    MoveInFreeSpaceOptimizer(const shared_ptr<MujocoHelper>& mujocoHelper,
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

    void setPointToMoveTo(double x, double y, double yaw) {
        _pointToMoveToX = x;
        _pointToMoveToY = y;
        _pointToMoveToYaw = yaw;
    }

private:
    bool isGoalAchieved(State state) override {
        return checkIfRobotIsNearThePoint(state);
    }

    bool checkIfRobotIsNearThePoint(State state) {
        setMuJoCoTo(state);
        double distance = getDistanceToPoint(_mujocoHelper);
        resetToInitialState();
        return distance < _REACHED_GOAL_THRESHOLD;
    }

    CostSequence cost(StateSequence stateSequence, ControlSequence controlSequence) override {
        return cost(stateSequence, controlSequence, _mujocoHelper);
    }

    CostSequence cost(StateSequence stateSequence, const ControlSequence& /* controlSequence */, const shared_ptr<MujocoHelper>& mujocoHelper) {
        if (!_movableObstacleNames)
            throw std::invalid_argument("You should set a list of movable obstacle names!");

        CostSequence costSequence(static_cast<unsigned int>(_n));
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

    double intermediateCosts(const State& currentState, const shared_ptr<MujocoHelper>& mujocoHelper) {
        double edge = edgeCost(currentState, mujocoHelper);
        double highForce = highForceToStaticObstacleCost(currentState, mujocoHelper);

        double collision = 0.0;
        if (_staticObstacleNames)
            collision = collisionCost(currentState, mujocoHelper);

        return edge + collision + highForce;
    }

    double getDistanceToPoint(const shared_ptr<MujocoHelper>& mujocoHelper) {
        double robotCurrentX = mujocoHelper->getRobotXpos();
        double robotCurrentY = mujocoHelper->getRobotYpos();

        double deltaX = _pointToMoveToX - robotCurrentX;
        double deltaY = _pointToMoveToY - robotCurrentY;

        return sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    double distanceToGoalCost(State currentState, const shared_ptr<MujocoHelper>& mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        double distanceOfGoalObjectToGoalRegion = getDistanceToPoint(mujocoHelper);

        if (distanceOfGoalObjectToGoalRegion < _REACHED_GOAL_THRESHOLD) {
            return 0.0;
        }

        return _REACHED_GOAL_WEIGHT * distanceOfGoalObjectToGoalRegion;
    }

    float edgeCost(State state, const shared_ptr<MujocoHelper>& mujocoHelper) {
        setMuJoCoTo(state, mujocoHelper);

        for (const auto & _movableObstacleName : *_movableObstacleNames) {
            double objectXpos = mujocoHelper->getBodyXpos(_movableObstacleName);
            double objectYpos = mujocoHelper->getBodyYpos(_movableObstacleName);
            double objectZpos = mujocoHelper->getBodyZpos(_movableObstacleName);

            if (objectZpos < _TABLE_Z)
                return _EDGE_COST;

            if (objectXpos > _PLANE_HIGH_X || objectXpos < _PLANE_LOW_X)
                return _EDGE_COST;

            if (objectYpos > _PLANE_HIGH_Y || objectYpos < _PLANE_LOW_Y)
                return _EDGE_COST;
        }

        return 0.0;
    }

    float highForceToStaticObstacleCost(State currentState, const shared_ptr<MujocoHelper>& mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);

        if (mujocoHelper->isHighForceAppliedToShelf()) {
            return _HIGH_FORCE_COST;
        }

        return 0.0;
    }

    float disturbanceCost(State currentState, State nextState, const shared_ptr<MujocoHelper>& mujocoHelper) {
        double costValue = 0.0;

        setMuJoCoTo(nextState, mujocoHelper);
        vector<tuple<double, double>> postPositions;
        for (const auto & _movableObstacleName : *_movableObstacleNames) {
            double objectXpos = mujocoHelper->getBodyXpos(_movableObstacleName);
            double objectYpos = mujocoHelper->getBodyYpos(_movableObstacleName);
            postPositions.push_back(make_tuple(objectXpos, objectYpos));
        }

        setMuJoCoTo(currentState, mujocoHelper);
        int index = 0;

        for (const auto & _movableObstacleName : *_movableObstacleNames) {
            double preObjectXpos = mujocoHelper->getBodyXpos(_movableObstacleName);
            double preObjectYpos = mujocoHelper->getBodyYpos(_movableObstacleName);

            double postObjectXpos = get<0>(postPositions[index]);
            double postObjectYpos = get<1>(postPositions[index]);

            double deltaX = postObjectXpos - preObjectXpos;
            double deltaY = postObjectYpos - preObjectYpos;
            costValue += sqrt(deltaX * deltaX + deltaY * deltaY);
            index++;
        }

        return _DISTURBANCE_WEIGHT * (costValue * costValue);
    }

    float collisionCost(State currentState, const shared_ptr<MujocoHelper>& mujocoHelper) {
        setMuJoCoTo(currentState, mujocoHelper);
        for (const auto & _staticObstacleName : *_staticObstacleNames) {
            if (mujocoHelper->isRobotInContact(_staticObstacleName)) {
                return _COLLISION_COST;
            }
        }

        return 0.0;
    }

    ControlSequence createCandidateControlSequence() override {
        double robotInitialX = _mujocoHelper->getRobotXpos();
        double robotInitialY = _mujocoHelper->getRobotYpos();

        double deltaX = _pointToMoveToX - robotInitialX;
        double deltaY = _pointToMoveToY - robotInitialY;
        double deltaYaw = _pointToMoveToYaw;

        double linearX = deltaX / _trajectoryDuration;
        double linearY = deltaY / _trajectoryDuration;
        double angularZ = deltaYaw;

        double currentGripperDOFValue = _mujocoHelper->getGripperDOFValue();
        Control velocities(linearX, linearY, 0.0, currentGripperDOFValue);
        ControlSequence controlSequence = ControlSequence(static_cast<unsigned int>(_n));
        for (int i = 0; i < _n - 1; ++i) {
            controlSequence.addControl(velocities);
        }

        // This step is to fix the orientation of the robot to be aligned
        // with the world, to make it easier for reaching through clutter
        // when the space is small.
        Control velocities2(0.0, 0.0, angularZ, currentGripperDOFValue);
        controlSequence.addControl(velocities2);

        resetToInitialState();

        return controlSequence;
    }

    const double _EDGE_COST = 500;
    const double _DISTURBANCE_WEIGHT = 0.000001;
    const double _COLLISION_COST = 500; // Any static obstacle collision.
    const double _REACHED_GOAL_WEIGHT = 5000;
    const double _HIGH_FORCE_COST = 500;
    const double _REACHED_GOAL_THRESHOLD = 0.07;

    // Plane limits
    const double _PLANE_HIGH_X;
    const double _PLANE_LOW_X;
    const double _PLANE_HIGH_Y;
    const double _PLANE_LOW_Y;
    const double _TABLE_Z;

    double _pointToMoveToX = 0.0;
    double _pointToMoveToY = 0.0;
    double _pointToMoveToYaw = 0.0;
};

#endif
