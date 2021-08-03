//  Copyright (C) 2018 Rafael Papallas and The University of Leeds
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

#include "mujoco.h"
#include <ompl/control/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include "glfw3.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <Eigen/Dense>
#include <utility>
#include "../src/State.cpp"
#include <boost/array.hpp>
//#include "kinematics/kinematics.cpp"

#ifndef MUJOCOHELPER
#define MUJOCOHELPER

namespace oc = ompl::control;
namespace ob = ompl::base;
using namespace std;

class MujocoHelper {
public:
    ~MujocoHelper() {
        mj_deleteModel(_model);
        mj_deleteData(_data);

        for (unsigned int i = 0; i < _mujocoStates.size(); ++i) {
            mj_deleteData(_mujocoStates[i]);
        }

        _mujocoStates.clear();
        _mujocoStates.shrink_to_fit();
    }

    MujocoHelper(mjModel *model, mjData *data, std::string robotName, std::string robotLinearXjointName,
                 std::string robotLinearYjointName, std::string robotAngularZjointName, const double robotInitialX,
                 const double robotInitialY) {
        _robotName = std::move(robotName);
        _robotLinearXjointName = std::move(robotLinearXjointName);
        _robotLinearYjointName = std::move(robotLinearYjointName);
        _robotAngularZjointName = std::move(robotAngularZjointName);
        _robotInitialXPosition = robotInitialX;
        _robotInitialYPosition = robotInitialY;

        _model = model;
        _data = data;

        init_other_attributes();
        _initialState = getState();
    }

    void setMovableObjectNames(vector<string> movableObjectNames) {
        _movableObjectNames = movableObjectNames;
    }

    vector<string> getMovableObjectNames() const {
        return _movableObjectNames;
    }

    State getLatestSavedState() const {
        return _latestState;
    }

    State getInitialState() const {
        return _initialState;
    }

    explicit MujocoHelper(const MujocoHelper *originalMujocoHelper) {
        _model = mj_copyModel(NULL, originalMujocoHelper->getModel());

        mjData *copiedData = mj_makeData(_model);
        mj_copyData(copiedData, _model, originalMujocoHelper->getData());
        _data = copiedData;

        _movableObjectNames = originalMujocoHelper->getMovableObjectNames();
        _ur5JointValues = originalMujocoHelper->getUR5JointValues();
        _ur5JointNames = originalMujocoHelper->getUR5JointNames();
        _latestState = originalMujocoHelper->getLatestSavedState();
        _initialState = originalMujocoHelper->getInitialState();

        _bestState = originalMujocoHelper->getBestState();
        _goalRegionX = originalMujocoHelper->getGoalRegionX();
        _goalRegionY = originalMujocoHelper->getGoalRegionY();

        _hasGripperDOF = originalMujocoHelper->isGripperEnabled();
        _model->opt.timestep = originalMujocoHelper->getModel()->opt.timestep;
        _gripperJointValues = originalMujocoHelper->getCurrentJointValuesForGripper();
        _gripperCurrentValue = originalMujocoHelper->getGripperDOFValue();

        _robotName = originalMujocoHelper->getRobotName();
        _robotLinearXjointName = originalMujocoHelper->getRobotLinearXjointName();
        _robotLinearYjointName = originalMujocoHelper->getRobotLinearYjointName();
        _robotAngularZjointName = originalMujocoHelper->getRobotAngularZjointName();
        _robotInitialXPosition = originalMujocoHelper->getRobotInitialXPosition();
        _robotInitialYPosition = originalMujocoHelper->getRobotInitialYPosition();

        _resetLevel = originalMujocoHelper->getResetLevel();
        for (auto data : originalMujocoHelper->getMujocoStates()) {
            mjData *copiedData2 = mj_makeData(_model);
            mj_copyData(copiedData2, _model, data);
            _mujocoStates.push_back(copiedData2);
        }

        init_other_attributes();
    }

    std::vector<mjData *> getMujocoStates() const {
        return _mujocoStates;
    }

    void init_other_attributes() {
        std::tuple<int, int> indicesLinearX = getJointQvelAddr(_robotLinearXjointName);
        std::tuple<int, int> indicesLinearY = getJointQvelAddr(_robotLinearYjointName);
        std::tuple<int, int> indicesAngularZ = getJointQvelAddr(_robotAngularZjointName);
        _robotLinearXjointQvelIndex = std::get<0>(indicesLinearX);
        _robotLinearYjointQvelIndex = std::get<0>(indicesLinearY);
        _robotAngularZjointQvelIndex = std::get<0>(indicesAngularZ);

        const std::string tableName = "table";
        _tableId = mj_name2id(_model, mjOBJ_BODY, tableName.c_str());
        _gripperJointNames = {"joint7_1", "joint8_1", "joint9_1", "joint10_1", "joint7_2", "joint8_2", "joint9_2",
                              "joint10_2"};
    }

    double getRobotInitialXPosition() const {
        return _robotInitialXPosition;
    }

    double getRobotInitialYPosition() const {
        return _robotInitialYPosition;
    }

    string getRobotName() const {
        return _robotName;
    }

    string getRobotLinearXjointName() const {
        return _robotLinearXjointName;
    }

    string getRobotLinearYjointName() const {
        return _robotLinearYjointName;
    }

    string getRobotAngularZjointName() const {
        return _robotAngularZjointName;
    }

    double getGoalRegionX() const {
        return _goalRegionX;
    }

    double getGoalRegionY() const {
        return _goalRegionY;
    }

    void enableGripperDOF() {
        _hasGripperDOF = true;
    }

    void disableGripperDOF() {
        _hasGripperDOF = false;
    }

    bool isGripperEnabled() const {
        return _hasGripperDOF;
    }


    // ===================================================================
    //                         MuJoCo Functions
    // ===================================================================


    void step() {
        forward();

        std::lock_guard<std::mutex> lockGuard(mtx);
        mj_step(_model, _data);
    }

    void forward() {
        std::lock_guard<std::mutex> lockGuard(mtx);

        if (_hasGripperDOF) {
            for (unsigned int i = 0; i < _gripperJointValues.size(); ++i) {
                std::tuple<int, int> jointAddr = getJointQposAddr(_gripperJointNames[i]);
                int jointIndex = std::get<0>(jointAddr);
                _data->qpos[jointIndex] = _gripperJointValues[i];
                std::tuple<int, int> jointVelAddr = getJointQvelAddr(_gripperJointNames[i]);
                int jointIndexForVel = std::get<0>(jointVelAddr);
                _data->qvel[jointIndexForVel] = 0.0;
            }
        }

//        for (unsigned int i = 0; i < _ur5JointNames.size(); ++i) {
//            auto jointAddress = getJointQposAddr(_ur5JointNames[i]);
//            int jointIndex = std::get<0>(jointAddress);
//            _data->qpos[jointIndex] = _ur5JointValues[i];
//            std::tuple<int, int> jointVelAddr = getJointQvelAddr(_ur5JointNames[i]);
//            int jointIndexForVel = std::get<0>(jointVelAddr);
//            _data->qvel[jointIndexForVel] = 0.0;
//        }

        mj_forward(_model, _data);
    }

    mjModel *getModel() const {
        return _model;
    }

    mjData *getData() const {
        if (_resetLevel > 0) {
            mjData *currentData = _mujocoStates.at(_resetLevel - 1);
            mj_copyData(_data, _model, currentData);
            return currentData;
        }

        return _data;
    }

    double *getNumericField(const std::string &fieldName) {
        for (int i = 0; i < _model->nnumeric; i++) {
            std::string f = _model->names + _model->name_numericadr[i];
            if (fieldName == f) {
                return _model->numeric_data + _model->numeric_adr[i];
            }
        }

        return nullptr;
    }

    string getTextField(const string &fieldName) {
        for (int i = 0; i < _model->ntext; i++) {
            std::string f = _model->names + _model->name_textadr[i];
            if (fieldName == f) {
                return _model->text_data + _model->text_adr[i];
            }
        }

        return string();

    }

    void setResetLevel(int level) {
        int resetLevelsAvailable = _mujocoStates.size();

        if (level > resetLevelsAvailable) {
            throw std::invalid_argument(
                    "The reset level is invalid. (there is/are " + std::to_string(resetLevelsAvailable) +
                    " only states you requested " + std::to_string(level) + ".)");
        } else if (level < 0) {
            throw std::invalid_argument("The reset level should be a positive number or 0.");
        }

        _resetLevel = level;
    }

    int getResetLevel() const {
        return _resetLevel;
    }

    void setResetLevelToLatest() {
        setResetLevel(getMaxResetLevel());
    }

    int getMaxResetLevel() {
        return _mujocoStates.size();
    }

    void saveMujocoState() {
        mjData *newData = mj_makeData(_model);
        mj_copyData(newData, _model, _data);
        _mujocoStates.push_back(newData);
    }

    void deleteLastState() {
        int n = getMaxResetLevel();
        if (n >= 1) {
            mj_deleteData(_mujocoStates[n - 1]);
            setResetLevel(n - 1);
        }
    }

    void emptyResetQueue() {
        for (unsigned int i = 0; i < _mujocoStates.size(); ++i) {
            mj_deleteData(_mujocoStates[i]);
        }

        _mujocoStates.clear();
        _mujocoStates.shrink_to_fit();

        setResetLevel(0);
        _latestState = _initialState;
    }

    void emptyResetQueueButLast() {
        mjData *latestData = mj_makeData(_model);
        mj_copyData(latestData, _model, _mujocoStates[_mujocoStates.size() - 1]);
        emptyResetQueue();
        _mujocoStates.push_back(latestData);
        setResetLevelToLatest();
    }

    void saveState() {
        saveMujocoState();
        _latestState = getState();
    }

    State getState() {
        State currentState;

        vector<tuple<double, double, double>> objectPositions;
        vector<tuple<double, double, double, double, double, double>> objectsVelocities;
        vector<tuple<double, double, double, double, double, double>> objectsAccelerations;
        vector<tuple<double, double, double, double, double, double>> objectsAccelerationWarmStart;

        for (unsigned int i = 0; i < _movableObjectNames.size(); ++i) {
            string objectName = _movableObjectNames[i];
            double objectXpos = getBodyXpos(objectName);
            double objectYpos = getBodyYpos(objectName);
            double objectYaw = getBodyYaw(objectName);
            objectPositions.push_back(make_tuple(objectXpos, objectYpos, objectYaw));

            auto objectVelocity = getBodyVelocity(objectName);
            objectsVelocities.push_back(objectVelocity);


            auto objectAccelerations = getBodyAccelerations(objectName);
            objectsAccelerations.push_back(objectAccelerations);

            auto objectAccelerationWarmStart = getBodyAccelerationWarmStart(objectName);
            objectsAccelerationWarmStart.push_back(objectAccelerationWarmStart);
        }

        currentState.setObjectPositions(objectPositions);
        currentState.setObjectVelocities(objectsVelocities);
        currentState.setObjectAccelerations(objectsAccelerations);
        currentState.setObjectAccelerationWarmStart(objectsAccelerationWarmStart);

        double robotXpos = getRobotXpos();
        double robotYpos = getRobotYpos();
        double robotYaw = getRobotYaw();
        tuple<double, double, double> robotPosition = make_tuple(robotXpos, robotYpos, robotYaw);
        currentState.setRobotPosition(robotPosition);

        double linearX = getRobotVelocity(0);
        double linearY = getRobotVelocity(1);
        double angularZ = getRobotVelocity(2);
        tuple<double, double, double> robotVelocity = make_tuple(linearX, linearY, angularZ);
        currentState.setRobotVelocity(robotVelocity);

        double gripperDOFValue = getGripperDOFValue();
        currentState.setGripperDOF(gripperDOFValue);

        currentState.setUR5Joints(_ur5JointValues);

        return currentState;
    }

    void restoreFrom(State& state) {
        if(state.isEmpty()) {
            return;
        }

        setUR5JointValues(state.getUr5Joints());

        auto robotPosition = state.getRobotPosition();
        auto robotVelocity = state.getRobotVelocity();

        setRobotXYPosition(get<0>(robotPosition), get<1>(robotPosition));
        setRobotYaw(get<2>(robotPosition));
        setRobotVelocity(get<0>(robotVelocity), get<1>(robotVelocity), get<2>(robotVelocity));

        auto objectPositions = state.getObjectPositions();
        auto objectVelocities = state.getObjectVelocities();
        auto objectAccelerations = state.getObjectAccelerations();
        auto objectAccelerationWarmStart = state.getObjectAccelerationWarmStart();

        if(_movableObjectNames.empty()) {
            printf("No movable object names set in MujocoHelepr!\n");
        }

        for (unsigned int i = 0; i < _movableObjectNames.size(); ++i) {
            string objectName = _movableObjectNames[i];
            setBodyXYPosition(objectName, get<0>(objectPositions[i]), get<1>(objectPositions[i]));
            setBodyYaw(objectName, get<2>(objectPositions[i]));

            double v1 = get<0>(objectVelocities[i]);
            double v2 = get<1>(objectVelocities[i]);
            double v3 = get<2>(objectVelocities[i]);
            double v4 = get<3>(objectVelocities[i]);
            double v5 = get<4>(objectVelocities[i]);
            double v6 = get<5>(objectVelocities[i]);
            setBodyVelocity(objectName, v1, v2, v3, v4, v5, v6);

            if(!objectAccelerations.empty()) {
                double a1 = get<0>(objectAccelerations[i]);
                double a2 = get<1>(objectAccelerations[i]);
                double a3 = get<2>(objectAccelerations[i]);
                double a4 = get<3>(objectAccelerations[i]);
                double a5 = get<4>(objectAccelerations[i]);
                double a6 = get<5>(objectAccelerations[i]);
                setBodyAccelerations(objectName, a1, a2, a3, a4, a5, a6);
            }

            if(!objectAccelerationWarmStart.empty()) {
                double a1 = get<0>(objectAccelerationWarmStart[i]);
                double a2 = get<1>(objectAccelerationWarmStart[i]);
                double a3 = get<2>(objectAccelerationWarmStart[i]);
                double a4 = get<3>(objectAccelerationWarmStart[i]);
                double a5 = get<4>(objectAccelerationWarmStart[i]);
                double a6 = get<5>(objectAccelerationWarmStart[i]);
                setBodyAccelerationWarmStart(objectName, a1, a2, a3, a4, a5, a6);
            }
        }

        setGripperDOFValue(state.getGripperDOFValue());
        forward();
//        step();
    }

    void resetSimulation() {
        std::unique_lock<mutex> lck(mtx);
        mj_resetData(_model, _data);
        lck.unlock();

        if (_resetLevel > 0) {
            mjData *targetData = _mujocoStates.at(_resetLevel - 1);
            mj_copyData(_data, _model, targetData);
        }

        restoreFrom(_latestState);

        forward();
        _isSystemValid = true;
    }

    void zeroOutVelocities() {
        for (int i = 0; i < _model->nv; ++i) {
            _data->qvel[i] = 0.0;
        }
    }

    double getTimeStep() {
        return _model->opt.timestep;
    }

    std::tuple<int, int> getJointQposAddr(const std::string &name) {
        int jointId = mj_name2id(_model, mjOBJ_JOINT, name.c_str());
        int jointType = _model->jnt_type[jointId];
        int jointAddr = _model->jnt_qposadr[jointId];

        int ndim;
        if (jointType == mjJNT_FREE)
            ndim = 7;
        else if (jointType == mjJNT_BALL)
            ndim = 4;
        else if (jointType == mjJNT_HINGE || jointType == mjJNT_SLIDE)
            ndim = 1;

        if (ndim == 1)
            return std::make_tuple(jointAddr, jointAddr);

        return std::make_tuple(jointAddr, jointAddr + ndim);
    }

    std::tuple<int, int> getJointQvelAddr(const std::string &jointName) {
        int jointId = mj_name2id(_model, mjOBJ_JOINT, jointName.c_str());
        int jointType = _model->jnt_type[jointId];
        int jointAddr = _model->jnt_dofadr[jointId];

        int ndim = 1;
        if (jointType == mjJNT_FREE)
            ndim = 6;
        else if (jointType == mjJNT_BALL)
            ndim = 3;
        else if (jointType == mjJNT_HINGE || jointType == mjJNT_SLIDE)
            ndim = 1;

        if (ndim == 1)
            return std::make_tuple(jointAddr, jointAddr);

        return std::make_tuple(jointAddr, jointAddr + ndim);
    }

    void setBodyVelocity(const std::string &name, double linearX, double linearY, double linearZ, double angularX,
                         double angularY, double angularZ) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int start = _model->jnt_dofadr[jointIndex];
        //auto indices = getJointQvelAddr(name);
        //int start = get<0>(indices);

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qvel[start + 0] = linearX;
        _data->qvel[start + 1] = linearY;
        _data->qvel[start + 2] = linearZ;
        _data->qvel[start + 3] = angularX;
        _data->qvel[start + 4] = angularY;
        _data->qvel[start + 5] = angularZ;
    }

    std::tuple<double, double, double, double, double, double> getBodyVelocity(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int start = _model->jnt_dofadr[jointIndex];
        //auto indices = getJointQvelAddr(name);
        //int start = get<0>(indices);

        std::lock_guard<std::mutex> lockGuard(mtx);
        double linearX = _data->qvel[start + 0];
        double linearY = _data->qvel[start + 1];
        double linearZ = _data->qvel[start + 2];
        double angularX = _data->qvel[start + 3];
        double angularY = _data->qvel[start + 4];
        double angularZ = _data->qvel[start + 5];

        return std::make_tuple(linearX, linearY, linearZ, angularX, angularY, angularZ);
    }

    void setModel(mjModel *model) {
        _model = model;
    }

    void setData(mjData *data) {
        _data = data;
        mj_resetData(_model, _data);
    }




    // ===================================================================
    //                         Gripper Functions
    // ===================================================================

    void printGripperJoints() {
        for (unsigned int i = 0; i < _gripperJointValues.size(); ++i) {
            std::tuple<int, int> jointaddr = getJointQposAddr(_gripperJointNames[i]);
            int jointindex = std::get<0>(jointaddr);
            double value = _data->qpos[jointindex];
            printf("%s: %f\n", _gripperJointNames[i].c_str(), value);
        }
    }

    static vector<double> getJointValuesForGripper(double value) {
        double joint7_1 = 0.822592 / 255.0 * value;
        double joint8_1 = -0.186819 / 255.0 * value;
        double joint9_1 = -0.642 / 255.0 * value;
        double joint10_1 = 0.721295 / 255.0 * value;

        double joint7_2 = 0.822595 / 255.0 * value;
        double joint8_2 = -0.186890 / 255.0 * value;
        double joint9_2 = -0.642 / 255.0 * value;
        double joint10_2 = 0.721342 / 255.0 * value;

        vector<double> gripperJointValues = {joint7_1, joint8_1, joint9_1, joint10_1, joint7_2, joint8_2, joint9_2,
                                             joint10_2};
        return gripperJointValues;
    }

    vector<double> getCurrentJointValuesForGripper() const {
        return _gripperJointValues;
    }

    void setGripperDOFValue(double value) {
        if (value > 255.0 || value < 0.0) {
            printf("The DOF value is: %f\n", value);
            throw std::invalid_argument(
                    "setGripperDOFValue: The gripper value should be in the range 0..255 inclusive.");
        }

        _gripperCurrentValue = value;
        vector<double> newJoints = getJointValuesForGripper(value);
        _gripperJointValues.assign(newJoints.begin(), newJoints.end()); // Method 1
    }

    double getGripperDOFValueFromJointValue() {
        double joint7_1 = _gripperJointValues[0];
        double value = joint7_1 * 255 / 0.822592;
        return int(value);
    }

    void setGripperJointValues(vector<double> jointValues) {
        vector<double> upperJointLimits = getJointValuesForGripper(255.0);
        vector<double> lowerJointLimits = getJointValuesForGripper(0.0);

        for (unsigned int i = 0; i < _gripperJointValues.size(); ++i) {
            _gripperJointValues[i] = jointValues[i];
        }

        _gripperCurrentValue = getGripperDOFValueFromJointValue();
    }

    double getGripperDOFValue() const {
        return _gripperCurrentValue;
    }

    // ===================================================================
    //                            Transformations
    // ===================================================================

    double getRollFromQuat(double w, double x, double y, double z) {
        //double sinr_cosp = +2.0 * (w * x + y * z);
        //double cosr_cosp = +1.0 - 2.0 * (x * x + y * y);
        //double roll = atan2(sinr_cosp, cosr_cosp);
        mj_normalizeQuat(_model, _data->qpos);

        double aSinInput = -2 * (x * z - w * y);
        if (aSinInput > 1.0)
            aSinInput = 1;
        if (aSinInput < -1.0)
            aSinInput = -1;

        return atan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z);
    }

    double getPitchFromQuat(double w, double x, double y, double z) {
        mj_normalizeQuat(_model, _data->qpos);

        double aSinInput = -2 * (x * z - w * y);
        if (aSinInput > 1.0)
            aSinInput = 1;
        if (aSinInput < -1.0)
            aSinInput = -1;

        return asin(aSinInput);
        //double sinp = +2.0 * (w * y - z * x);
        //double pitch;
        //if (fabs(sinp) >= 1)
        //pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        //else
        //pitch = asin(sinp);

        //return pitch;
    }

    double getYawFromQuat(double w, double x, double y, double z) {
        mj_normalizeQuat(_model, _data->qpos);

        double aSinInput = -2 * (x * z - w * y);
        if (aSinInput > 1.0)
            aSinInput = 1;
        if (aSinInput < -1.0)
            aSinInput = -1;

        return atan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z);
        //double siny_cosp = +2.0 * (w * z + x * y);
        //double cosy_cosp = +1.0 - 2.0 * (y * y + z * z);
        //double yaw = atan2(siny_cosp, cosy_cosp);

        //return yaw;
    }

    static boost::array<double, 4> getQuatFromAxisAngle(mjtNum *axis, double value) {
        mjtNum quaternion[4];
        mju_axisAngle2Quat(quaternion, axis, value);

        boost::array<double, 4> finalQuaternion = {{quaternion[0], quaternion[1], quaternion[2], quaternion[3]}};
        return finalQuaternion;
    }

    boost::array<double, 4> getQuatFromRoll(double roll) {
        mjtNum xaxis[3] = {1, 0, 0};
        return getQuatFromAxisAngle(xaxis, roll);
    }

    boost::array<double, 4> getQuatFromPitch(double pitch) {
        mjtNum yaxis[3] = {0, 1, 0};
        return getQuatFromAxisAngle(yaxis, pitch);
    }

    boost::array<double, 4> getQuatFromYaw(double yaw) {
        mjtNum zaxis[3] = {0, 0, 1};
        return getQuatFromAxisAngle(zaxis, yaw);
    }

    static boost::array<double, 4> getQuatFromTransform(Eigen::MatrixXf transform) {
        double m00 = transform(0, 0);
        double m01 = transform(0, 1);
        double m02 = transform(0, 2);
        double m10 = transform(1, 0);
        double m11 = transform(1, 1);
        double m12 = transform(1, 2);
        double m20 = transform(2, 0);
        double m21 = transform(2, 1);
        double m22 = transform(2, 2);

        double tr = m00 + m11 + m22;

        double qw, qx, qy, qz;
        if (tr > 0) {
            float S = sqrt(tr + 1.0) * 2; // S=4*qw
            qw = 0.25 * S;
            qx = (m21 - m12) / S;
            qy = (m02 - m20) / S;
            qz = (m10 - m01) / S;
        } else if ((m00 > m11) & (m00 > m22)) {
            float S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
            qw = (m21 - m12) / S;
            qx = 0.25 * S;
            qy = (m01 + m10) / S;
            qz = (m02 + m20) / S;
        } else if (m11 > m22) {
            float S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
            qw = (m02 - m20) / S;
            qx = (m01 + m10) / S;
            qy = 0.25 * S;
            qz = (m12 + m21) / S;
        } else {
            float S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
            qw = (m10 - m01) / S;
            qx = (m02 + m20) / S;
            qy = (m12 + m21) / S;
            qz = 0.25 * S;
        }

        boost::array<double, 4> quaternion = {{qw, qx, qy, qz}};
        return quaternion;
    }


    // ===================================================================
    //                            Bodies
    // ===================================================================

    mjtGeom getGeomTypeFromBodyName(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        int bodyNumberOfGeoms = _model->body_geomnum[bodyId];

        // If the body has more geoms than 1 or no geoms then it's
        // ambigious and we throw an error.
        if (bodyNumberOfGeoms != 1) {
            throw std::invalid_argument(
                    "bodyGeomType: The given body name should have exactly 1 geom defined (" + name + ").");
        }

        int addrOfBodyGeom = _model->body_geomadr[bodyId];
        return (mjtGeom) _model->geom_type[addrOfBodyGeom];
    }

    double getBodyXpos(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        return _data->qpos[qposIndex + 0];
    }

    double getBodyYpos(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        return _data->qpos[qposIndex + 1];
    }

    double getBodyZpos(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        return _data->qpos[qposIndex + 2];
    }

    double getBodyRoll(std::string name) {
        return getBodyEulerAngle(name, 0);
    }

    double getBodyPitch(std::string name) {
        return getBodyEulerAngle(name, 1);
    }

    double getBodyYaw(std::string name) {
        return getBodyEulerAngle(name, 2);
    }

    double getBodyEulerAngle(const std::string &name, int axis) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        double w = _data->qpos[qposIndex + 3];
        double x = _data->qpos[qposIndex + 4];
        double y = _data->qpos[qposIndex + 5];
        double z = _data->qpos[qposIndex + 6];

        switch (axis) {
            case 0 :
                return getRollFromQuat(w, x, y, z);
            case 1 :
                return getPitchFromQuat(w, x, y, z);
            case 2 :
                return getYawFromQuat(w, x, y, z);
            default:
                return 0.0;
        }
    }

    void setBodyRoll(const std::string &name, double roll) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        boost::array<double, 4> quaternion = getQuatFromRoll(roll);
        double w = quaternion[0];
        double x = quaternion[1];
        double y = quaternion[2];
        double z = quaternion[3];

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 3] = w;
        _data->qpos[qposIndex + 4] = x;
        _data->qpos[qposIndex + 5] = y;
        _data->qpos[qposIndex + 6] = z;
    }

    std::tuple<double, double, double, double> getBodyQuaternion(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        double w = _data->qpos[qposIndex + 3];
        double x = _data->qpos[qposIndex + 4];
        double y = _data->qpos[qposIndex + 5];
        double z = _data->qpos[qposIndex + 6];

        return std::make_tuple(w, x, y, z);
    }

    void setBodyPitch(const std::string &name, double pitch) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        boost::array<double, 4> quaternion = getQuatFromPitch(pitch);
        double w = quaternion[0];
        double x = quaternion[1];
        double y = quaternion[2];
        double z = quaternion[3];

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 3] = w;
        _data->qpos[qposIndex + 4] = x;
        _data->qpos[qposIndex + 5] = y;
        _data->qpos[qposIndex + 6] = z;
    }

    void setBodyYaw(const std::string &name, double yaw) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        boost::array<double, 4> quaternion = getQuatFromYaw(yaw);
        double w = quaternion[0];
        double x = quaternion[1];
        double y = quaternion[2];
        double z = quaternion[3];

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 3] = w;
        _data->qpos[qposIndex + 4] = x;
        _data->qpos[qposIndex + 5] = y;
        _data->qpos[qposIndex + 6] = z;
    }

    void setBodyQuat(const std::string &name, double w, double x, double y, double z) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 3] = w;
        _data->qpos[qposIndex + 4] = x;
        _data->qpos[qposIndex + 5] = y;
        _data->qpos[qposIndex + 6] = z;
    }

    void setBodyXYPosition(const std::string &name, double x, double y) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);

        _data->qpos[qposIndex + 0] = x;
        _data->qpos[qposIndex + 1] = y;
    }

    void setBodyZPosition(const std::string &name, double z) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 2] = z;
    }

    double getBodyVelocity(std::string jointName, int desiredVelocityIndex) {
        std::tuple<int, int> indices = getJointQvelAddr(jointName);
        int startIndex = std::get<0>(indices);
        int endIndex = std::get<1>(indices);

        if (startIndex + desiredVelocityIndex > endIndex) {
            throw std::invalid_argument("getBodyVelocity: Index out of bounds.");
        }

        return _data->qvel[startIndex + desiredVelocityIndex];
    }

    void setBodyQuatToNull(const std::string &bodyName) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];

        double w = 1.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 3] = w;
        _data->qpos[qposIndex + 4] = x;
        _data->qpos[qposIndex + 5] = y;
        _data->qpos[qposIndex + 6] = z;
    }

    void setBodyAccelerations(const std::string &bodyName, double a1, double a2, double a3, double a4, double a5,
                              double a6) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qvelAdr = _model->jnt_dofadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qacc[qvelAdr + 0] = a1;
        _data->qacc[qvelAdr + 1] = a2;
        _data->qacc[qvelAdr + 2] = a3;
        _data->qacc[qvelAdr + 3] = a4;
        _data->qacc[qvelAdr + 4] = a5;
        _data->qacc[qvelAdr + 5] = a6;
    }

    std::tuple<double, double, double, double, double, double> getBodyAccelerations(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qvelAdr = _model->jnt_dofadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        double a1 = _data->qacc[qvelAdr + 0];
        double a2 = _data->qacc[qvelAdr + 1];
        double a3 = _data->qacc[qvelAdr + 2];
        double a4 = _data->qacc[qvelAdr + 3];
        double a5 = _data->qacc[qvelAdr + 4];
        double a6 = _data->qacc[qvelAdr + 5];

        return std::make_tuple(a1, a2, a3, a4, a5, a6);
    }

    void setBodyAccelerationWarmStart(const std::string &bodyName, double a1, double a2, double a3, double a4, double a5, double a6) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qvelAdr = _model->jnt_dofadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qacc_warmstart[qvelAdr + 0] = a1;
        _data->qacc_warmstart[qvelAdr + 1] = a2;
        _data->qacc_warmstart[qvelAdr + 2] = a3;
        _data->qacc_warmstart[qvelAdr + 3] = a4;
        _data->qacc_warmstart[qvelAdr + 4] = a5;
        _data->qacc_warmstart[qvelAdr + 5] = a6;
    }

    std::tuple<double, double, double, double, double, double> getBodyAccelerationWarmStart(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qvelAdr = _model->jnt_dofadr[jointIndex];

        std::lock_guard<std::mutex> lockGuard(mtx);
        double a1 = _data->qacc_warmstart[qvelAdr + 0];
        double a2 = _data->qacc_warmstart[qvelAdr + 1];
        double a3 = _data->qacc_warmstart[qvelAdr + 2];
        double a4 = _data->qacc_warmstart[qvelAdr + 3];
        double a5 = _data->qacc_warmstart[qvelAdr + 4];
        double a6 = _data->qacc_warmstart[qvelAdr + 5];

        return std::make_tuple(a1, a2, a3, a4, a5, a6);
    }

    Eigen::MatrixXf getBodyTransform(const std::string &name) {
        // Create an identity matrix
        Eigen::MatrixXf transform = Eigen::MatrixXf::Identity(4, 4);

        // Get robot's theta value.
        double theta = getBodyYaw(name);
        double x = getBodyXpos(name);
        double y = getBodyYpos(name);
        double z = getBodyZpos(name);

        // Create the array such that it looks like this:
        //  cos(θ)    sin(θ)   0     x
        // -sin(θ)    cos(θ)   0     y
        //  0         0        1     z
        //  0         0        0     1

        transform(0, 0) = cos(theta);
        transform(0, 1) = -sin(theta);
        transform(1, 0) = sin(theta);
        transform(1, 1) = cos(theta);

        transform(0, 3) = x;
        transform(1, 3) = y;
        transform(2, 3) = z;

        return transform;
    }

    bool isHighForceAppliedToShelf() {
        return isHighForceAppliedToBody("shelf", 9);
    }

    bool isHighForceAppliedToBody(std::string bodyName, double threshold) {
        vector<mjtNum *> bodyForceTorques = getBodyForceTorques(bodyName);
        for (mjtNum *forceTorque : bodyForceTorques) {
            for (int i = 0; i < 3; ++i) { // 3 forces
                if (forceTorque[i] >= threshold)
                    return true;
            }
        }

        return false;
    }

    vector<mjtNum *> getBodyForceTorques(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());

        std::lock_guard<std::mutex> lockGuard(mtx);
        int numberOfContacts = _data->ncon;

        vector<int> contactIds;
        for (int i = 0; i < numberOfContacts; ++i) {
            auto contact = _data->contact[i];

            int bodyInContact1 = _model->body_rootid[_model->geom_bodyid[contact.geom1]];
            int bodyInContact2 = _model->body_rootid[_model->geom_bodyid[contact.geom2]];

            bool tableIsNotInContact = bodyInContact1 != _tableId && bodyInContact2 != _tableId;
            bool desiredBodyIsInContact = bodyInContact1 == bodyId || bodyInContact2 == bodyId;

            if (tableIsNotInContact && desiredBodyIsInContact) {
                contactIds.push_back(i);
            }
        }

        vector<mjtNum *> forceTorques = std::vector<mjtNum *>();
        if (!contactIds.empty()) {
            for (int contactId : contactIds) {
                mjtNum bodyForceTorque[6];
                mj_contactForce(_model, _data, contactId, bodyForceTorque);
                forceTorques.push_back(bodyForceTorque);
            }
        }

        return forceTorques;
    }

    std::tuple<double, double, double> getBodyTorque(const std::string &name) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, name.c_str());

        std::lock_guard<std::mutex> lockGuard(mtx);
        //mjtNum* bodyForceTorque;
        //mj_contactForce(_model, _data, bodyId, bodyForceTorque);
        //double t1 = bodyForceTorque[3];
        //double t2 = bodyForceTorque[4];
        //double t3 = bodyForceTorque[5];
        //return std::make_tuple(t1, t2, t3);

        //std::lock_guard<std::mutex> lockGuard(mtx);
        double t1 = _data->xfrc_applied[(6 * bodyId) + 3];
        double t2 = _data->xfrc_applied[(6 * bodyId) + 4];
        double t3 = _data->xfrc_applied[(6 * bodyId) + 5];
        return std::make_tuple(t1, t2, t3);
    }

    void setJoint(const string &name, double desiredValue) {
        auto jointAddress = getJointQposAddr(name);
        int jointIndex = std::get<0>(jointAddress);
        for (unsigned int i = 0; i < _ur5JointNames.size(); ++i) {
            if (_ur5JointNames[i] == name) {
                _ur5JointValues[i] = desiredValue;
                std::lock_guard<std::mutex> lockGuard(mtx);
                _data->qpos[jointIndex] = desiredValue;
                break;
            }
        }
    }

    void setUR5JointsToZero() {
        for (unsigned int i = 0; i < _ur5JointNames.size(); ++i) {
            auto jointAddress = getJointQposAddr(_ur5JointNames[i]);
            int jointIndex = std::get<0>(jointAddress);
            _ur5JointValues[i] = 0.0;
            _data->qpos[jointIndex] = 0.0;
        }
    }

    vector<double> getUR5JointValues() const {
        return _ur5JointValues;
    }

    vector<string> getUR5JointNames() const {
        return _ur5JointNames;
    }

    void setUR5JointValues(vector<double> jointValues) {
        _ur5JointValues = jointValues;
    }

    // ===================================================================
    //                                 Robot
    // ===================================================================

    void updateRobotInitialXPosition(double initialX) {
        _robotInitialXPosition = initialX;
    }

    void updateRobotInitialYPosition(double initialY) {
        _robotInitialYPosition = initialY;
    }

    double getRobotXpos() {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, _robotName.c_str());

        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        return _robotInitialXPosition + _data->qpos[qposIndex + 0];
    }

    double dotProduct(vector<double> vect_A, vector<double> vect_B) const {
        double product = 0.0;

        // Loop for calculate cot product
        for (unsigned int i = 0; i < vect_A.size(); i++)
            product = product + vect_A[i] * vect_B[i];

        return product;
    }

    static double normOf(const vector<double> &vector) {
        double sum = 0.0;
        for (auto value : vector) {
            sum += value * value;
        }

        return sqrt(sum);
    }

    vector<double> unitVectorOf(const vector<double> &vector) {
        double norm = normOf(vector);
        std::vector<double> unitVector;
        for (auto value : vector) {
            unitVector.push_back(value / norm);
        }

        return unitVector;
    }

    double getDistanceOfEndEffectorTo(const string &objectName) {
        // Current position of goal object.
        double goalObjectX = getBodyXpos(objectName);
        double goalObjectY = getBodyYpos(objectName);

        auto endEffectorPosition = getSitePosition("ee_point_1"); // This is the position of the end-effector.
        double endEffectorX = endEffectorPosition[0];
        double endEffectorY = endEffectorPosition[1];

        auto site2 = getSitePosition("ee_point_2");
        double site2_x = site2[0];
        double site2_y = site2[1];

        // Direction vector
        double eeVectorX = site2_x - endEffectorX;
        double eeVectorY = site2_y - endEffectorY;
        vector<double> directionVector = {eeVectorX, eeVectorY};
        vector<double> unitDirectionVector = unitVectorOf(directionVector);

        // Find the vector of the end-effector to the goal object.
        double eeToGoalX = goalObjectX - endEffectorX;
        double eeToGoalY = goalObjectY - endEffectorY;
        vector<double> eeToGoalVector = {eeToGoalX, eeToGoalY};
        vector<double> unitEeToGoalVector = unitVectorOf(eeToGoalVector);

        const double PHI = 0.09; // Angles are larger numbers (distances in meters)
        double angle = PHI * acos(dotProduct(unitDirectionVector, unitEeToGoalVector));


//        double x1, x2, y1, y2;
//        x1 = unitDirectionVector[0];
//        y1 = unitDirectionVector[1];
//
//        x2 = unitEeToGoalVector[0];
//        y2 = unitEeToGoalVector[1];
//
//        double angle = PHI * (atan2(y2, x2) - atan2(y1, x1));


//        double angle = atan2(unitEeToGoalVector[1], unitEeToGoalVector[0]) - atan2(unitDirectionVector[1], unitDirectionVector[0]);
//        if (angle > M_PI)        { angle -= 2 * M_PI; }
//        else if (angle <= -M_PI) { angle += 2 * M_PI; }
//        angle *= PHI;

        return sqrt(eeToGoalX * eeToGoalX + eeToGoalY * eeToGoalY + angle * angle);
    }

    vector<double> getSitePosition(const std::string &siteName) {
        int siteID = mj_name2id(_model, mjOBJ_SITE, siteName.c_str());

        double x = _data->site_xpos[3 * siteID + 0];
        double y = _data->site_xpos[3 * siteID + 1];
        double z = _data->site_xpos[3 * siteID + 2];

        vector<double> values = {x, y, z};
        return values;
    }

    Eigen::MatrixXf getSiteTransform(std::string siteName) {
        vector<double> sitePosition = getSitePosition(siteName);

        // Site Position
        Eigen::MatrixXf transform = Eigen::MatrixXf::Identity(4, 4);
        transform(0, 3) = sitePosition[0];
        transform(1, 3) = sitePosition[1];
        transform(2, 3) = sitePosition[2];

        int siteID = mj_name2id(_model, mjOBJ_SITE, siteName.c_str());
        double m0 = _data->site_xmat[9 * siteID + 0];
        double m1 = _data->site_xmat[9 * siteID + 1];
        double m2 = _data->site_xmat[9 * siteID + 2];
        double m3 = _data->site_xmat[9 * siteID + 3];
        double m4 = _data->site_xmat[9 * siteID + 4];
        double m5 = _data->site_xmat[9 * siteID + 5];
        double m6 = _data->site_xmat[9 * siteID + 6];
        double m7 = _data->site_xmat[9 * siteID + 7];
        double m8 = _data->site_xmat[9 * siteID + 8];

        transform(0, 0) = m0;
        transform(0, 1) = m1;
        transform(0, 2) = m2;

        transform(1, 0) = m3;
        transform(1, 1) = m4;
        transform(1, 2) = m5;

        transform(2, 0) = m6;
        transform(2, 1) = m7;
        transform(2, 2) = m8;

        return transform;
    }

    double getRobotYpos() {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, _robotName.c_str());

        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        return _robotInitialYPosition + _data->qpos[qposIndex + 1];
    }

    // TODO: We assume that the robot is planar here, needs to be fixed when we moved away from the planar robot.
    static double getRobotZpos() {
        return 0.0;
    }

    Eigen::MatrixXf getRobotTransform() {
        // Create an identity matrix
        Eigen::MatrixXf transform = Eigen::MatrixXf::Identity(4, 4);

        // Get robot's theta value.
        double theta = getRobotYaw();
        double x = getRobotXpos();
        double y = getRobotYpos();
        double z = getRobotZpos();

        // Create the array such that it looks like this:
        //  cos(θ)    sin(θ)   0     x
        // -sin(θ)    cos(θ)   0     y
        //  0         0        1     z
        //  0         0        0     1

        transform(0, 0) = cos(theta);
        transform(0, 1) = -sin(theta);
        transform(1, 0) = sin(theta);
        transform(1, 1) = cos(theta);

        transform(0, 3) = x;
        transform(1, 3) = y;
        transform(2, 3) = z;

        return transform;
    }

    Eigen::MatrixXf getEndEffectorInRobotTransform() {
        auto robotInWorldTransform = getRobotTransform();
        auto endEffectorInWorldTransform = getEndEffectorTransform();
        Eigen::MatrixXf endEffectorInRobotTransform = robotInWorldTransform.inverse() * endEffectorInWorldTransform;
        return endEffectorInRobotTransform;
    }

    Eigen::MatrixXf getEndEffectorTransform() {
        return getSiteTransform("ee_point_1");
    }

    tuple<double, double> getEndEffectorXYpositionFromRobotState(double x, double y, double yaw) {
        // Create an identity matrix
        Eigen::MatrixXf robotInWorld = Eigen::MatrixXf::Identity(4, 4);

        // Create the array such that it looks like this:
        //  cos(θ)    sin(θ)   0     x
        // -sin(θ)    cos(θ)   0     y
        //  0         0        1     z
        //  0         0        0     1
        robotInWorld(0, 0) = cos(yaw);
        robotInWorld(0, 1) = -sin(yaw);
        robotInWorld(1, 0) = sin(yaw);
        robotInWorld(1, 1) = cos(yaw);

        robotInWorld(0, 3) = x;
        robotInWorld(1, 3) = y;
        robotInWorld(2, 3) = getRobotZpos();

        Eigen::MatrixXf endEffectorInRobot = getEndEffectorInRobotTransform();

        Eigen::MatrixXf endEffectorInWorld = robotInWorld * endEffectorInRobot;

        return make_tuple(endEffectorInWorld(0, 3), endEffectorInWorld(1, 3));
    }

    void setRobotXYPosition(double x, double y) {
        x = x - _robotInitialXPosition;
        y = y - _robotInitialYPosition;

        int bodyId = mj_name2id(_model, mjOBJ_BODY, _robotName.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);

        _data->qpos[qposIndex + 0] = x;
        _data->qpos[qposIndex + 1] = y;
    }

    void setRobotYaw(double yaw) {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, _robotName.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qpos[qposIndex + 2] = yaw;
    }

    double getRobotYaw() {
        int bodyId = mj_name2id(_model, mjOBJ_BODY, _robotName.c_str());
        const int jointIndex = _model->body_jntadr[bodyId];
        const int qposIndex = _model->jnt_qposadr[jointIndex];
        std::lock_guard<std::mutex> lockGuard(mtx);
        return _data->qpos[qposIndex + 2];
    }


    // ===================================================================
    //                             Controllers
    // ===================================================================

    void setRobotVelocity(double linearX, double linearY, double angularZ) {
        std::lock_guard<std::mutex> lockGuard(mtx);
        _data->qvel[_robotLinearXjointQvelIndex] = linearX;
        _data->qvel[_robotLinearYjointQvelIndex] = linearY;
        _data->qvel[_robotAngularZjointQvelIndex] = angularZ;
    }

    double getRobotVelocity(int index) {
        std::lock_guard<std::mutex> lockGuard(mtx);
        switch (index) {
            case 0:
                return _data->qvel[_robotLinearXjointQvelIndex];
                break;
            case 1:
                return _data->qvel[_robotLinearYjointQvelIndex];
                break;
            case 2:
                return _data->qvel[_robotAngularZjointQvelIndex];
                break;
            default:
                throw std::invalid_argument("In getRobotVelocity: Index out of bounds.");
        }
    }

    void
    propagateThreadSafe(const ob::State *state, const oc::Control *control, const double duration, ob::State *result,
                        const vector<string> *_movableObjects) {
        std::lock_guard<std::mutex> lockGuard(propagateMtx);
        const vector<string> &movableObjects = *_movableObjects;

        resetSimulation();

        // Controls
        const double *controls = control->as<oc::RealVectorControlSpace::ControlType>()->values;

        // Robot state state prior propagation.
        const ob::CompoundStateSpace::StateType *s = state->as<ob::CompoundStateSpace::StateType>();
        const ob::SE2StateSpace::StateType *robotState = s->as<ob::SE2StateSpace::StateType>(0);

        // Set robot's start state.
        setRobotXYPosition(robotState->getX(), robotState->getY());
        setRobotYaw(robotState->getYaw());

        // Set objects' start state.
        for (unsigned int i = 0; i < movableObjects.size(); ++i) {
            std::string movableObjectName = movableObjects[i];

            // i + 1 because subspaces for movable objects are 1..n and 0
            // is the robot subspace.
            const double *q = s->as<ob::RealVectorStateSpace::StateType>(i + 1)->values;
            setBodyXYPosition(movableObjectName, q[0], q[1]);
            setBodyYaw(movableObjectName, q[2]);
            setBodyVelocity(movableObjectName, q[3], q[4], q[5], q[6], q[7], q[8]);
            setBodyAccelerations(movableObjectName, q[9], q[10], q[11], q[12], q[13], q[14]);
        }

        //_mujocoHelper->setAllWarmstart(0.0);
        step();

        bool isSystemValid = true;

        // Propagate
        // The for loop will loop the required number of steps to complete
        // a propagation of a given duration. 1.0s / 0.001 gives 1000 steps.
        for (int i = 0; i < duration / getTimeStep(); ++i) {
            setRobotVelocity(controls[0], controls[1], controls[2]);
            step();

            if (i % 200 == 0 && isHighForceAppliedToShelf()) {
                isSystemValid = false;
                break;
            }
        }

        //for(int i=0; i < duration / getTimeStep(); ++i) {
        //step();
        //}

        // Robot resulted state after propagation.
        ob::CompoundStateSpace::StateType *res = result->as<ob::CompoundStateSpace::StateType>();
        ob::SE2StateSpace::StateType *resultedRobotState = res->as<ob::SE2StateSpace::StateType>(0);

        // Get the result of the state space after propagation.
        resultedRobotState->setX(getRobotXpos());
        resultedRobotState->setY(getRobotYpos());
        resultedRobotState->setYaw(getRobotYaw());

        // and now for the objects.
        for (unsigned int i = 0; i < movableObjects.size(); ++i) {
            std::string movableObjectName = movableObjects[i];

            // i + 1 because subspaces for movable objects are 1..n and 0
            // is the robot subspace.
            double *resultedObjectState = res->as<ob::RealVectorStateSpace::StateType>(i + 1)->values;
            resultedObjectState[0] = getBodyXpos(movableObjectName);
            resultedObjectState[1] = getBodyYpos(movableObjectName);
            resultedObjectState[2] = getBodyYaw(movableObjectName);

            std::tuple<double, double, double, double, double, double> velocities = getBodyVelocity(movableObjectName);
            resultedObjectState[3] = std::get<0>(velocities);
            resultedObjectState[4] = std::get<1>(velocities);
            resultedObjectState[5] = std::get<2>(velocities);
            resultedObjectState[6] = std::get<3>(velocities);
            resultedObjectState[7] = std::get<4>(velocities);
            resultedObjectState[8] = std::get<5>(velocities);

            // INFO: Note that accelerations now are set to 0.0 regardless of these values (see MujocoHelper).
            std::tuple<double, double, double, double, double, double> acceleration = getBodyAccelerations(
                    movableObjectName);
            resultedObjectState[9] = std::get<0>(acceleration);
            resultedObjectState[10] = std::get<1>(acceleration);
            resultedObjectState[11] = std::get<2>(acceleration);
            resultedObjectState[12] = std::get<3>(acceleration);
            resultedObjectState[13] = std::get<4>(acceleration);
            resultedObjectState[14] = std::get<5>(acceleration);

            double roll = getBodyRoll(movableObjectName);
            double pitch = getBodyPitch(movableObjectName);

            if (roll > 0.1 || roll < -0.1)
                isSystemValid = false;
            if (pitch > 0.1 || pitch < -0.1)
                isSystemValid = false;
        }

        if (!isSystemValid) {
            // Change a state to be clearly out of bounds to "fake" that we don't have a valid system.
            resultedRobotState->setX(1000);
        }
    }


    // ===================================================================
    //                         Collision Checking
    // ===================================================================

    void setContactAttributesForAllGeomsInBody(const string &bodyName, int contype, int conaffinity) {
        const int bodyId = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());
        const int numberOfGeomsInBody = _model->body_geomnum[bodyId];

        const int bodyGeomStartingAddress = _model->body_geomadr[bodyId];

        for (int offset = 0; offset < numberOfGeomsInBody; ++offset) {
            _model->geom_contype[bodyGeomStartingAddress + offset] = contype;
            _model->geom_conaffinity[bodyGeomStartingAddress + offset] = conaffinity;
        }
    }

    void disableCollisionsFor(const string &bodyName) {
        setContactAttributesForAllGeomsInBody(bodyName, 4, 4);
        forward();
        step();
    }

    void enableCollisionsFor(const string &bodyName) {
        setContactAttributesForAllGeomsInBody(bodyName, 3, 3);
        forward();
        step();
    }

//    vector<IkReal> matrixToVector(Eigen::MatrixXf matrix) {
//        vector<IkReal> pose;
//        pose.push_back(matrix(0, 0));
//        pose.push_back(matrix(0, 1));
//        pose.push_back(matrix(0, 2));
//        pose.push_back(matrix(0, 3));
//        pose.push_back(matrix(1, 0));
//        pose.push_back(matrix(1, 1));
//        pose.push_back(matrix(1, 2));
//        pose.push_back(matrix(1, 3));
//        pose.push_back(matrix(2, 0));
//        pose.push_back(matrix(2, 1));
//        pose.push_back(matrix(2, 2));
//        pose.push_back(matrix(2, 3));
//
//        return pose;
//    }

    vector<vector<double>> ik(Eigen::MatrixXf desiredEndEffectorTransform) {
        printf("Please ensure that end_effector_helper body is in the world!\n");
        double x = desiredEndEffectorTransform(0, 3);
        double y = desiredEndEffectorTransform(1, 3);

        auto quat = getQuatFromTransform(desiredEndEffectorTransform);
        setBodyQuat("end_effector_helper", quat[0], quat[1], quat[2], quat[3]);

        double originalX = getBodyXpos("end_effector_helper");
        double originalY = getBodyYpos("end_effector_helper");
        setBodyXYPosition("end_effector_helper", x, y);
        forward();

        desiredEndEffectorTransform = getBodyTransform("end_effector_helper");

        Eigen::MatrixXf baseLinkTransform = getSiteTransform("base_site");
        Eigen::MatrixXf localTransform = baseLinkTransform.inverse() * desiredEndEffectorTransform;

//        auto pose = matrixToVector(localTransform);
//        vector<float> result = _ikFast.inverse(pose);
        vector<double> result1;

        setBodyXYPosition("end_effector_helper", originalX, originalY);
        forward();

        vector<vector<double>> solutions;
//        for (unsigned int i = 0; i < result.size(); i += 6) {
//            vector<double> solution;
//
//            for (int j = 0; j < 6; ++j) {
//                solution.push_back(result[i + j]);
//            }
//
//            solutions.push_back(solution);
//        }

        return solutions;
    }

    Eigen::MatrixXf calculateEndEffectorTransformFromRobotState(double x, double y, double yaw) {
        Eigen::MatrixXf transform = Eigen::MatrixXf::Identity(4, 4);

        transform(0, 3) = x;
        transform(1, 3) = y;

        transform(0, 0) = cos(yaw);
        transform(0, 1) = -sin(yaw);
        transform(1, 0) = sin(yaw);
        transform(1, 1) = cos(yaw);

        Eigen::MatrixXf endEffectorInRobot = getEndEffectorInRobotTransform();
        Eigen::MatrixXf endEffectorInWorld = transform * endEffectorInRobot;

        return endEffectorInWorld;
    }

    int getBodyIdFromGeomId(int geomId) {
        return _model->geom_bodyid[geomId];
    }

    bool isBodyInContact(const std::string &bodyName) {
        std::lock_guard<std::mutex> lockGuard(mtx);
        const int bodyId = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());
        int numberOfContacts = _data->ncon;

        for (int i = 0; i < numberOfContacts; ++i) {
            auto contact = _data->contact[i];

            int bodyInContact1 = _model->body_rootid[_model->geom_bodyid[contact.geom1]];
            int bodyInContact2 = _model->body_rootid[_model->geom_bodyid[contact.geom2]];

            bool tableIsNotInContact = bodyInContact1 != _tableId && bodyInContact2 != _tableId;
            bool bodyIsInContact = bodyInContact1 == bodyId || bodyInContact2 == bodyId;

            if (tableIsNotInContact && bodyIsInContact)
                return true;
        }

        return false;
    }

    bool isBodyInContact(const std::string &bodyName1, const std::string &bodyName2) {
        std::lock_guard<std::mutex> lockGuard(mtx);
        const int bodyId1 = mj_name2id(_model, mjOBJ_BODY, bodyName1.c_str());
        const int bodyId2 = mj_name2id(_model, mjOBJ_BODY, bodyName2.c_str());

        int numberOfContacts = _data->ncon;

        for (int i = 0; i < numberOfContacts; ++i) {
            auto contact = _data->contact[i];

            int bodyInContact1 = _model->body_rootid[_model->geom_bodyid[contact.geom1]];
            int bodyInContact2 = _model->body_rootid[_model->geom_bodyid[contact.geom2]];

            bool isBody1InContact = bodyInContact1 == bodyId1 || bodyInContact2 == bodyId1;
            bool isBody2InContact = bodyInContact1 == bodyId2 || bodyInContact2 == bodyId2;

            if (isBody1InContact && isBody2InContact)
                return true;
        }

        return false;
    }

    bool isRobotInContact(const std::string &bodyName) {
        std::lock_guard<std::mutex> lockGuard(mtx);
        int bodyId1 = mj_name2id(_model, mjOBJ_BODY, _robotName.c_str());
        const int bodyId2 = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());

        int numberOfContacts = _data->ncon;

        for (int i = 0; i < numberOfContacts; ++i) {
            auto contact = _data->contact[i];

            int bodyInContact1 = _model->body_rootid[_model->geom_bodyid[contact.geom1]];
            int bodyInContact2 = _model->body_rootid[_model->geom_bodyid[contact.geom2]];

            bool isBody1InContact = bodyInContact1 == bodyId1 || bodyInContact2 == bodyId1;
            bool isBody2InContact = bodyInContact1 == bodyId2 || bodyInContact2 == bodyId2;

            if (isBody1InContact && isBody2InContact)
                return true;
        }

        return false;
    }

    bool isRobotInContact() {
        return isBodyInContact(_robotName);
    }

    std::tuple<double, double, double>
    getRobotStateFromEndEffectorTransform(const Eigen::MatrixXf &endEffectorInWorld) {
        // Convert endEffectorTransform to RobotTransform
        Eigen::MatrixXf endEffectorInRobot = getEndEffectorInRobotTransform();
        Eigen::MatrixXf robotInEndEffector = endEffectorInRobot.inverse();
        Eigen::MatrixXf robotInWorld = endEffectorInWorld * robotInEndEffector;
        double x = robotInWorld(0, 3);
        double y = robotInWorld(1, 3);

        // Convert Transformation Matrix into Quaternion.
        mjtNum target_transform[9];
        target_transform[0] = robotInWorld(0, 0);
        target_transform[1] = robotInWorld(0, 1);
        target_transform[2] = robotInWorld(0, 2);
        target_transform[3] = robotInWorld(1, 0);
        target_transform[4] = robotInWorld(1, 1);
        target_transform[5] = robotInWorld(1, 2);
        target_transform[6] = robotInWorld(2, 0);
        target_transform[7] = robotInWorld(2, 1);
        target_transform[8] = robotInWorld(2, 2);

        mjtNum quat[4];
        mju_mat2Quat(quat, target_transform);

        // Extract Yaw from Quaternion.
        double yaw = getYawFromQuat(quat[0], quat[1], quat[2], quat[3]);

        return std::make_tuple(x, y, yaw);
    }

    bool checkRobotTransformForCollisions(const Eigen::MatrixXf &transformToCheck,
                                          const vector<string> &objectNamesToCheckForCollisions) {

        for (const string &objectName : objectNamesToCheckForCollisions) {
            if (checkRobotTransformForCollisions(transformToCheck, objectName)) {
                return true;
            }
        }

        return false;
    }

    bool checkRobotTransformForCollisions(const Eigen::MatrixXf &transformToCheck,
                                          string objectNameToCheckForCollisions) {
        //std::lock_guard<std::mutex> lockGuard(mtx);
        tuple<double, double, double> robotState = getRobotStateFromEndEffectorTransform(transformToCheck);
        double x = get<0>(robotState);
        double y = get<1>(robotState);
        double yaw = get<2>(robotState);

        int originalResetLevel = _resetLevel;

        setRobotXYPosition(x, y);
        setRobotYaw(yaw);
        step();

        bool isInCollision = isRobotInContact(objectNameToCheckForCollisions);

        setResetLevel(originalResetLevel);

        // Reset the system to the previous state
        resetSimulation();

        return isInCollision;
    }

    int getBodyNumberOfCollisions(const string &bodyName) {
        std::lock_guard<std::mutex> lockGuard(mtx);
        int bodyId = mj_name2id(_model, mjOBJ_BODY, bodyName.c_str());

        int numberOfContacts = _data->ncon;
        int numberOfCollisions = 0;
        for (int i = 0; i < numberOfContacts; ++i) {
            auto contact = _data->contact[i];

            int bodyInContact1 = _model->body_rootid[_model->geom_bodyid[contact.geom1]];
            int bodyInContact2 = _model->body_rootid[_model->geom_bodyid[contact.geom2]];

            bool theDesiredBodyIsInContact = bodyInContact1 == bodyId || bodyInContact2 == bodyId;
            bool theOtherBodyIsNotTheTable = bodyInContact1 != _tableId && bodyInContact2 != _tableId;

            if (theDesiredBodyIsInContact && theOtherBodyIsNotTheTable)
                numberOfCollisions++;
        }

        return numberOfCollisions;
    }

    int getRobotNumberOfCollisions() {
        return getBodyNumberOfCollisions(_robotName);
    }

    int getNumberOfCollisionsForRobotsDesiredState(double x, double y, double yaw) {
        int originalResetLevel = _resetLevel;

        setRobotXYPosition(x, y);
        setRobotYaw(yaw);
        step();

        int numberOfCollisions = getRobotNumberOfCollisions();

        setResetLevel(originalResetLevel);

        // Reset the system to the previous state
        resetSimulation();

        return numberOfCollisions;
    }


    // ===================================================================
    //                         System Methods
    // ===================================================================


    bool isSystemValid() {
        return _isSystemValid;
    }

    void isSystemValid(bool isValid) {
        _isSystemValid = isValid;
    }

    void setGoalRegion(double x, double y) {
        _goalRegionX = x;
        _goalRegionY = y;
    }

    void saveState(ob::State *state) {
        //std::lock_guard<std::mutex> lockGuard(mtx);
        const ob::CompoundStateSpace::StateType *st = state->as<ob::CompoundStateSpace::StateType>();
        const ob::SE2StateSpace::StateType *robotState = st->as<ob::SE2StateSpace::StateType>(0);
        const double *goalObjectState = st->as<ob::RealVectorStateSpace::StateType>(1)->values;

        double robotX = robotState->getX();
        double robotY = robotState->getY();
        double robotYaw = robotState->getYaw();

        Eigen::MatrixXf transform = Eigen::MatrixXf::Identity(4, 4);

        transform(0, 0) = cos(robotYaw);
        transform(0, 1) = -sin(robotYaw);
        transform(1, 0) = sin(robotYaw);
        transform(1, 1) = cos(robotYaw);

        transform(0, 3) = robotX;
        transform(1, 3) = robotY;
        transform(2, 3) = 0.0;

        Eigen::MatrixXf endEffectorInRobot = getEndEffectorInRobotTransform();
        Eigen::MatrixXf endEffectorInWorld = transform * endEffectorInRobot;

        double ee_x = endEffectorInWorld(0, 3);
        double ee_y = endEffectorInWorld(1, 3);
        double goal_x = goalObjectState[0];
        double goal_y = goalObjectState[1];

        double dx = ee_x - goal_x;
        double dy = ee_y - goal_y;

        double distance = sqrt(dx * dx + dy * dy);

        if (distance < _bestStateDistance) {
            _bestStateDistance = distance;
            _bestState = state;
        }
    }

    ompl::base::State *getBestState() const {
        return _bestState;
    }

private:
    mjModel *_model;
    mjData *_data;
    std::mutex mtx;
    std::mutex propagateMtx;
    double _bestStateDistance = 100000;
    ob::State *_bestState;
    bool _isSystemValid = true; // This is used in StatePropagator to check if an object has rolled or pitched.
    double _goalRegionX;
    double _goalRegionY;
    int _resetLevel = 0;
    std::vector<mjData *> _mujocoStates;
    int _tableId;

    string _robotName;
    std::string _robotLinearXjointName;
    int _robotLinearXjointQvelIndex;

    std::string _robotLinearYjointName;
    int _robotLinearYjointQvelIndex;

    std::string _robotAngularZjointName;
    int _robotAngularZjointQvelIndex;

    double _robotInitialXPosition;
    double _robotInitialYPosition;

    // gripper
    bool _hasGripperDOF = false;
    vector<double> _gripperJointValues = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double _gripperCurrentValue = 0.0;
    vector<string> _gripperJointNames;
    vector<int> _bodyIdsToIgnoreCollisions;

    vector<double> _ur5JointValues = {-1.494190, -2.847766, -1.497504, -1.952727, -1.546158, -3.14159};
//    vector<double> _ur5JointValues = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    vector<string> _ur5JointNames = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                     "wrist_2_joint", "wrist_3_joint"};

    vector<string> _movableObjectNames;
    State _latestState;
    State _initialState;
//    Kinematics _ikFast;
};

#endif
