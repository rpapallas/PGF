#include <vector>
using std::vector;

class Control {
public:
    Control(double linearX, double linearY, double angularZ, double gripperDOF) {
        _linearX = linearX;
        _linearY = linearY;
        _angularZ = angularZ;
        _gripperDOF = gripperDOF;

        if (_gripperDOF > 255.0 || _gripperDOF < 0.0) {
            throw std::invalid_argument("The gripper DOF value is out of bounds.");
        }
    }

    double getLinearX() {
        return _linearX;
    }

    double getLinearY() {
        return _linearY;
    }

    double getAngularZ() {
        return _angularZ;
    }

    double getGripperDOF() {
        return _gripperDOF;
    }

    double getLinearXNoise() {
        return _linearXnoise;
    }

    double getLinearYNoise() {
        return _linearYnoise;
    }

    double getAngularZNoise() {
        return _angularZnoise;
    }

    double getGripperDOFNoise() {
        return _gripperDOFnoise;
    }

    void setLinearX(double x) {
        _linearX = x;
    }

    void setLinearY(double y) {
        _linearY = y;
    }

    void setAngularZ(double z) {
        _angularZ = z;
    }

    void setLinearXnoise(double noise) {
        _linearXnoise = noise;
        _linearX += _linearXnoise;
    }

    void setLinearYnoise(double noise) {
        _linearYnoise = noise;
        _linearY += _linearYnoise;
    }

    void setAngularZnoise(double noise) {
        _angularZnoise = noise;
        _angularZ += _angularZnoise;
    }

    void setGripperDOFnoise(double noise) {
        if (_gripperDOF + noise > 255.0 || _gripperDOF + noise < 0.0) {
            throw std::invalid_argument("You are attempting to add noise to a control that is out of bounds.");
        }

        _gripperDOFnoise = noise;
        _gripperDOF += _gripperDOFnoise;
    }

    bool isNull() {
        return _linearX == 0 && _linearY == 0 && _angularZ == 0 && _gripperDOF == 0;
    }

    bool isNotNull() {
        return !isNull();
    }

    void setIK(vector<double> ikSolution) {
        _ikSolution = ikSolution;
    }

    vector<double> getIkSolution() {
        return _ikSolution;
    }

    bool ikSolutionExists() {
        return !_ikSolution.empty();
    }

private:
    double _linearX;
    double _linearY;
    double _angularZ;
    double _gripperDOF;
    double _linearXnoise = 0.0;
    double _linearYnoise = 0.0;
    double _angularZnoise = 0.0;
    double _gripperDOFnoise = 0.0;
    vector<double> _ikSolution;
};
