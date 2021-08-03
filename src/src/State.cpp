#ifndef MUJOCOSTATE
#define MUJOCOSTATE

#include <vector>

using namespace std;

class State {
public:
    State() = default;

    explicit State(const State *originalState) {
        _objectPositions = originalState->getObjectPositions();
        _objectVelocities = originalState->getObjectVelocities();
        _objectAccelerations = originalState->getObjectAccelerations();
        _robotPosition = originalState->getRobotPosition();
        _robotVelocity = originalState->getRobotVelocity();
        _gripperDOFValue = originalState->getGripperDOFValue();
        _UR5Joints = originalState->getUr5Joints();
    }

    void setObjectPositions(vector<tuple<double, double, double>> objectPositions) {
        _objectPositions = objectPositions;
    }

    void setObjectVelocities(vector<tuple<double, double, double, double, double, double>> objectVelocities) {
        _objectVelocities = objectVelocities;
    }

    void setObjectAccelerations(vector<tuple<double, double, double, double, double, double>> accelerations) {
        _objectAccelerations = accelerations;
    }

    void setObjectAccelerationWarmStart(vector<tuple<double, double, double, double, double, double>> accelerationWarmstart) {
        _objectAccelerationWarmStart = accelerationWarmstart;
    }

    void setRobotPosition(tuple<double, double, double> robotPosition) {
        _robotPosition = robotPosition;
    }

    void setRobotVelocity(tuple<double, double, double> robotVelocity) {
        _robotVelocity = robotVelocity;
    }

    void setGripperDOF(double value) {
        _gripperDOFValue = value;
    }

    void setUR5Joints(vector<double> joints) {
        _UR5Joints = joints;
    }

    vector<tuple<double, double, double>> getObjectPositions() const {
        return _objectPositions;
    }

    vector<tuple<double, double, double, double, double, double>> getObjectVelocities() const {
        return _objectVelocities;
    }

    vector<tuple<double, double, double, double, double, double>> getObjectAccelerations() const {
        return _objectAccelerations;
    }

    vector<tuple<double, double, double, double, double, double>> getObjectAccelerationWarmStart() const {
        return _objectAccelerationWarmStart;
    }

    tuple<double, double, double> getRobotPosition() const {
        return _robotPosition;
    }

    tuple<double, double, double> getRobotVelocity() const {
        return _robotVelocity;
    }

    double getGripperDOFValue() const {
        return _gripperDOFValue;
    }

    vector<double> getUr5Joints() const {
        return _UR5Joints;
    }

    bool isEmpty() {
        return _objectPositions.empty();
    }

private:
    vector<double> _UR5Joints;
    vector<tuple<double, double, double>> _objectPositions;
    vector<tuple<double, double, double, double, double, double>> _objectVelocities;
    vector<tuple<double, double, double, double, double, double>> _objectAccelerations;
    vector<tuple<double, double, double, double, double, double>> _objectAccelerationWarmStart;
    tuple<double, double, double> _robotPosition;
    tuple<double, double, double> _robotVelocity;
    double _gripperDOFValue = 0.0;
};

#endif
