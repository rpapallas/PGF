#ifndef TRAJOPT
#define TRAJOPT

#include <future>
#include <mutex>
#include <utility>
#include "State.cpp"
#include "StateSequence.cpp"
#include "Control.cpp"
#include "ControlSequence.cpp"
#include "CostSequence.cpp"
#include "Result.cpp"
#include "../utils/MujocoHelper.cpp"
#include "ros/ros.h"
#include "pgf/PredictionInput.h"
#include "pgf/RobotBid.h"
#include "Prediction.cpp"


class OptimizerBase {
public:
    explicit OptimizerBase(const shared_ptr<MujocoHelper> &mujocoHelper) {
        _mujocoHelper = make_shared<MujocoHelper>(mujocoHelper.get());
        updateActionDuration();
    }

    virtual Result optimize() {
        _result = NullResult(0.0);
        _isOptimizing = true;

        if (!_samplingVarianceVector) {
            throw std::runtime_error("The variance vector parameter was not set. Quiting");
        }

        auto optimizationStart = std::chrono::high_resolution_clock::now();

        // This is an initial candidate assuming to be the best so far.
        auto initialControlSequenceAndCost = getInitialControlSequenceAndItsCost();
        ControlSequence bestControlSequence = get<0>(initialControlSequenceAndCost);
        double bestCost = get<2>(initialControlSequenceAndCost).sum();

        if (bestControlSequence.size() == 0) {
            printf("\033[0;31mNot able to generate a valid, initial control sequence. Optimisation aborted.\033[0m\n");
            return NullResult(0);
        }

        // We set _result from now so we can allow someone externally to visualize the current best trajectory.
        Result initialResult(false, 0.0, bestControlSequence);
        _result = initialResult;

        int iterations = 0;
        bool maxIterationNotReached = iterations < _maxIterations;
        bool costThresholdNotReached = bestCost > _costThreshold;

        while (maxIterationNotReached && costThresholdNotReached) {
            iterations++;

            resetToInitialState();
            auto rollouts = createNoisyTrajectories(bestControlSequence);
            resetToInitialState();

            ControlSequence currentControlSequence = updateTrajectory(bestControlSequence, rollouts);

            auto currentRollout = trajectoryRollout(currentControlSequence);
            auto currentCost = get<1>(currentRollout).sum();

            printf("%d. Prior Cost: %f | Updated Cost: %f\n", iterations, bestCost, currentCost);

            if (currentCost < bestCost) {
                bestControlSequence = currentControlSequence;
                bestCost = currentCost;

                // We set _result from now so we can allow someone externally to visualize the current best trajectory.
                Result currentResult(false, 0.0, bestControlSequence);
                _result = currentResult;
            }

            costThresholdNotReached = bestCost > _costThreshold;
            maxIterationNotReached = iterations < _maxIterations;
        }

        resetToInitialState();

        auto optimizationFinish = std::chrono::high_resolution_clock::now();
        double optimizationElapsedTime = formatDurationToSeconds(optimizationStart, optimizationFinish);
        printf("\033[0;33mOptimization took:\033[0m %f seconds after %d iterations.\n", optimizationElapsedTime,
               iterations);

        auto bestStateSequence = get<0>(trajectoryRollout(bestControlSequence));
        State finalState = bestStateSequence.getState(bestStateSequence.size() - 1);
        bool didReachTheGoal = isGoalAchieved(finalState);

        Result finalResult(didReachTheGoal, optimizationElapsedTime, bestControlSequence);
        _result = finalResult;
        _isOptimizing = false;
        return finalResult;
    }

    tuple<ControlSequence, StateSequence, CostSequence> getInitialControlSequenceAndItsCost() {
        ControlSequence initialControlSequence = createCandidateControlSequence();

        auto rollout = trajectoryRollout(initialControlSequence);
        auto stateSequence = get<0>(rollout);
        auto costSequence = get<1>(rollout);

        return make_tuple(initialControlSequence, stateSequence, costSequence);
    }

    // Getters

    double getActionDuration() {
        updateActionDuration();
        return _actionDuration;
    }

    bool isOptimizing() {
        return _isOptimizing;
    }

    bool isProbabilisticUpdate() {
        return _USE_PROBABILISTIC_UPDATE;
    }

    // Setters

    void setInitialState(State initialState) {
        _initialState = initialState;
    }

    void setMovableObstacleNames(vector<string> *obstacleNames) {
        _movableObstacleNames = obstacleNames;
    }

    void setStaticObstacleNames(vector<string> *obstacleNames) {
        _staticObstacleNames = obstacleNames;
    }

    void setMaxIterations(int maxIterations) {
        _maxIterations = maxIterations;
    }

    void setNumberOfNoisyTrajectoryRollouts(int numberOfNoisyTrajectoryRollouts) {
        _numberOfNoisyTrajectoryRollouts = numberOfNoisyTrajectoryRollouts;

        // Create a MuJoCo Helper for each of the K noisy trajectories (for palatalization).
        createLocalMujocoHelpers();
    }

    void setCostThreshold(double costThreshold) {
        _costThreshold = costThreshold;
    }

    void setSamplingVarianceVector(vector<double> *samplingVarianceVector) {
        _samplingVarianceVector = samplingVarianceVector;
    }

    void setTrajectoryDuration(double duration) {
        _trajectoryDuration = duration;
        updateActionDuration();
    }

    void setControlSequenceSteps(int n) {
        _n = n;
        updateActionDuration();
    }

    void resetToInitialState(const shared_ptr<MujocoHelper> mujocoHelper) {
        mujocoHelper->resetSimulation();
    }

protected:
    // Virtual Methods

    virtual bool isGoalAchieved(State) = 0;

    virtual CostSequence cost(StateSequence, ControlSequence) = 0;

    virtual ControlSequence createCandidateControlSequence() = 0;

    // Utilities

    void resetToInitialState() {
        resetToInitialState(_mujocoHelper);
    }

    double formatDurationToSeconds(std::chrono::high_resolution_clock::time_point start,
                                   std::chrono::high_resolution_clock::time_point end) {
        int milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        return milliseconds / 1000.0;
    }

    State executeControlSequenceAndObtainState(const ControlSequence &controlSequence,
                                               const shared_ptr<MujocoHelper> &mujocoHelper) {
        for (Control currentControl : controlSequence.getControls()) {
            if(currentControl.isNull())
                continue;

            vector<double> initialGripperJointValues = mujocoHelper->getCurrentJointValuesForGripper();
            vector<double> finalGripperJointValues = mujocoHelper->getJointValuesForGripper(
                    currentControl.getGripperDOF()); // Fully-closed

            double steps = _actionDuration / mujocoHelper->getTimeStep();

            vector<double> diffGripperJointValues = finalGripperJointValues;
            for (unsigned int i = 0; i < initialGripperJointValues.size(); ++i) {
                diffGripperJointValues[i] -= initialGripperJointValues[i];
            }

            for (int step = 0; step < steps; ++step) {
                mujocoHelper->setRobotVelocity(currentControl.getLinearX(), currentControl.getLinearY(),
                                               currentControl.getAngularZ());

                // Set gripper DOF
                vector<double> stepGripperJointValues = initialGripperJointValues;
                for (unsigned int i = 0; i < stepGripperJointValues.size(); ++i) {
                    stepGripperJointValues[i] += diffGripperJointValues[i] * (step / steps);
                }

                mujocoHelper->setGripperJointValues(stepGripperJointValues);

                mujocoHelper->step();
            }
        }

        State state = getCurrentStateFromMujoco(mujocoHelper);
        resetToInitialState();
        return state;
    }

    void updateActionDuration() {
        _actionDuration = _trajectoryDuration / _n;
    }

    void setMuJoCoTo(State &state, const shared_ptr<MujocoHelper> &mujocoHelper) {
        mujocoHelper->restoreFrom(state);
        mujocoHelper->forward();
    }

    void setMuJoCoTo(State &state) {
        setMuJoCoTo(state, _mujocoHelper);
    }

    State getCurrentStateFromMujoco(const shared_ptr<MujocoHelper> &mujocoHelper) {
        return mujocoHelper->getState();
    }

    void createLocalMujocoHelpers(const shared_ptr<MujocoHelper> &mujocoHelper) {
        _mujocoHelpers.clear();
        _mujocoHelpers.shrink_to_fit();

        for (int k = 0; k < _numberOfNoisyTrajectoryRollouts; k++) {
            _mujocoHelpers.push_back(make_shared<MujocoHelper>(mujocoHelper.get()));
        }
    }

    void createLocalMujocoHelpers() {
        createLocalMujocoHelpers(_mujocoHelper);
    }

    // Trajectory Update

    static ControlSequence greedyUpdate(vector<tuple<StateSequence, ControlSequence, CostSequence>> rollouts) {
        ControlSequence minControlSequence = get<1>(rollouts[0]);
        double minCost = get<2>(rollouts[0]).sum();

        for (unsigned int i = 1; i < rollouts.size(); i++) {
            double currentCost = get<2>(rollouts[i]).sum();
            if (currentCost < minCost) {
                minCost = currentCost;
                minControlSequence = get<1>(rollouts[i]);
            }
        }

        return minControlSequence;
    }

    ControlSequence probabilisticUpdate(ControlSequence controlSequence,
                                        vector<tuple<StateSequence, ControlSequence, CostSequence>> rollouts) {
        unsigned int n = get<2>(rollouts[0]).size();
        const double h = 10;

        ControlSequence newControlSequence(_n);

        for (unsigned int i = 0; i < n; ++i) {
            double maxCost = std::numeric_limits<double>::min();
            double minCost = std::numeric_limits<double>::max();

            for (auto rollout : rollouts) {
                // k
                CostSequence costSequence = get<2>(rollout);
                double currentCost = costSequence.getCost(i);
                if (currentCost < minCost)
                    minCost = currentCost;
                if (currentCost > maxCost)
                    maxCost = currentCost;
            }

            double denom = maxCost - minCost;
            if (denom < 1e-8)
                denom = 1e-8;

            double pSum = 0.0;
            for (auto rollout : rollouts) {
                CostSequence costSequence = get<2>(rollout);
                double currentCost = costSequence.getCost(i);
                double prob = exp(-h * (currentCost - minCost) / denom);
                pSum += prob;
            }

            double noiseLinearX = 0.0;
            double noiseLinearY = 0.0;
            double noiseAngularZ = 0.0;
            double noiseGripperDOFValue = 0.0;

            double linearX = controlSequence.getControl(i).getLinearX();
            double linearY = controlSequence.getControl(i).getLinearY();
            double angularZ = controlSequence.getControl(i).getAngularZ();
            double gripperDOFValue = controlSequence.getControl(i).getGripperDOF();

            for (auto rollout : rollouts) {
                CostSequence costSequence = get<2>(rollout);
                double currentCost = costSequence.getCost(i);
                double prob = exp(-h * (currentCost - minCost) / denom);
                prob = prob / pSum;
                ControlSequence controlSeq = get<1>(rollout);
                noiseLinearX += prob * controlSeq.getControl(i).getLinearXNoise();
                noiseLinearY += prob * controlSeq.getControl(i).getLinearYNoise();
                noiseAngularZ += prob * controlSeq.getControl(i).getAngularZNoise();

                if (_samplingVarianceVector->size() >= 3) {
                    noiseGripperDOFValue += prob * controlSeq.getControl(i).getGripperDOFNoise();

                    // make sure that the noise + the previous value won't make
                    // an invalid dof value.
                    if (gripperDOFValue + noiseGripperDOFValue > 255.0 ||
                        gripperDOFValue + noiseGripperDOFValue < 0.0) {
                        noiseGripperDOFValue = 0.0;
                    }
                }
            }

            Control control(linearX, linearY, angularZ, gripperDOFValue);
            control.setLinearXnoise(noiseLinearX);
            control.setLinearYnoise(noiseLinearY);
            control.setAngularZnoise(noiseAngularZ);
            control.setGripperDOFnoise(noiseGripperDOFValue);

            newControlSequence.addControl(control);
        }

        return newControlSequence;
    }

    ControlSequence updateTrajectory(ControlSequence controlSequence,
                                     const vector<tuple<StateSequence, ControlSequence, CostSequence>> &rollouts) {
        if (_USE_PROBABILISTIC_UPDATE)
            return probabilisticUpdate(std::move(controlSequence), rollouts);

        return greedyUpdate(rollouts);
    }

    // Noisy Trajectory Generation

    ControlSequence createNoisyTrajectoryFrom(ControlSequence controlSequence) {
        ControlSequence noisyControlSequence(_n);

        for (unsigned int j = 0; j < controlSequence.size(); ++j) {
            Control previousValue = controlSequence.getControl(j);

            double linearX = previousValue.getLinearX();
            double linearY = previousValue.getLinearY();
            double angularZ = previousValue.getAngularZ();
            double gripperDOF = previousValue.getGripperDOF();
            double linearXnoise = 0.0;
            double linearYnoise = 0.0;
            double angularZnoise = 0.0;
            double gripperDOFnoise = 0.0;

            std::random_device rd;
            std::mt19937 gen(rd());


            vector<double> gripperDOFs = {0.0, 100, 150, 255.0};
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist6(0, gripperDOFs.size() - 1);

            for (unsigned int i = 0; i < _samplingVarianceVector->size(); ++i) {
                std::normal_distribution<double> d(0.0, _samplingVarianceVector->at(i));
                double noise = d(gen);

                if (i == 0) {
                    linearXnoise = noise;
                } else if (i == 1) {
                    linearYnoise = noise;
                } else if (i == 2) {
                    angularZnoise = noise;
                } else if (i == 3) { // Gripper DOF
                    double randomGripperDOF = gripperDOFs[dist6(rng)];
                    if (gripperDOF >= randomGripperDOF)
                        gripperDOFnoise = -(gripperDOF - randomGripperDOF);
                    else
                        gripperDOFnoise = (randomGripperDOF - gripperDOF);
                } else {
                    throw std::invalid_argument(
                            "You are attempting to sample a noise for more DOF than the DOFs of the system.");
                }
            }

            Control noisyControl(linearX, linearY, angularZ, gripperDOF);
            noisyControl.setLinearXnoise(linearXnoise);
            noisyControl.setLinearYnoise(linearYnoise);
            noisyControl.setAngularZnoise(angularZnoise);
            noisyControl.setGripperDOFnoise(gripperDOFnoise);
            noisyControlSequence.addControl(noisyControl);
        }

        return noisyControlSequence;
    }

    tuple<StateSequence, ControlSequence, CostSequence>
    generateNoisyTrajectoryThreaded(ControlSequence &controlSequence, int k) {
        auto mujocoHelper = _mujocoHelpers[k];
        resetToInitialState(mujocoHelper);
        auto noisyControlSequence = createNoisyTrajectoryFrom(controlSequence);
        auto rollout = trajectoryRollout(noisyControlSequence, mujocoHelper);
        auto stateSequence = get<0>(rollout);
        auto costSequence = get<1>(rollout);

        auto fullRollout = make_tuple(stateSequence, noisyControlSequence, costSequence);
        return fullRollout;
    }

    vector<tuple<StateSequence, ControlSequence, CostSequence>>
    createNoisyTrajectories(ControlSequence controlSequence) {

#pragma omp declare reduction (merge : std::vector<tuple<StateSequence, ControlSequence, CostSequence>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
        vector<tuple<StateSequence, ControlSequence, CostSequence>> rollouts;
#pragma omp parallel for reduction(merge: rollouts)
        for (int k = 0; k < _numberOfNoisyTrajectoryRollouts; ++k) {
            rollouts.push_back(generateNoisyTrajectoryThreaded(controlSequence, k));
        }


        return rollouts;

    }

    tuple<StateSequence, CostSequence> trajectoryRollout(const ControlSequence &controls) {
        return trajectoryRollout(controls, _mujocoHelper);
    }

    tuple<StateSequence, CostSequence>
    trajectoryRollout(const ControlSequence &controls, const shared_ptr<MujocoHelper> &mujocoHelper) {
        State currentState = _initialState;
        setMuJoCoTo(currentState, mujocoHelper);

        StateSequence stateSequence(_n);
        double steps = _actionDuration / mujocoHelper->getTimeStep();
		vector<string> objectNamesInCollision;
        for (Control currentControl : controls.getControls()) {
            if(currentControl.isNull()) {
                continue;
            }

            vector<double> initialGripperJointValues = mujocoHelper->getCurrentJointValuesForGripper();
            vector<double> finalGripperJointValues = mujocoHelper->getJointValuesForGripper(
                    currentControl.getGripperDOF()); // Fully-closed


            vector<double> diffGripperJointValues = finalGripperJointValues;
            for (unsigned int i = 0; i < initialGripperJointValues.size(); ++i) {
                diffGripperJointValues[i] -= initialGripperJointValues[i];
            }

            for (int step = 0; step < steps; ++step) {
                mujocoHelper->setRobotVelocity(currentControl.getLinearX(), currentControl.getLinearY(),
                                               currentControl.getAngularZ());

                // Set gripper DOF
                vector<double> stepGripperJointValues = initialGripperJointValues;
                for (unsigned int i = 0; i < stepGripperJointValues.size(); ++i) {
                    stepGripperJointValues[i] += diffGripperJointValues[i] * (step / steps);
                }

                mujocoHelper->setGripperJointValues(stepGripperJointValues);
                mujocoHelper->step();
            }

			for (unsigned int i = 0; i < _movableObstacleNames->size(); ++i) {
				string obj = _movableObstacleNames->at(i);
				if(std::find(objectNamesInCollision.begin(), objectNamesInCollision.end(), obj) == objectNamesInCollision.end()) {
					if(obj != "object_3" && mujocoHelper->isRobotInContact(obj)) {
						objectNamesInCollision.push_back(obj);
					}
				}
			}

            currentState = getCurrentStateFromMujoco(mujocoHelper);
            stateSequence.addState(currentState);
        }

        CostSequence costSequence = cost(stateSequence, controls);
		costSequence.setNumberOfObjectsInCollision(objectNamesInCollision.size());
        return make_tuple(stateSequence, costSequence);
    }

    shared_ptr<MujocoHelper> _mujocoHelper;
    State _initialState;
    int _numberOfNoisyTrajectoryRollouts = 8;
    int _maxIterations = 100;
    double _costThreshold = 100.0;
    double _trajectoryDuration = 10.0;
    int _n = 50;
    double _actionDuration = 1.0;
    std::vector<double> *_samplingVarianceVector = nullptr;
    std::vector<string> *_movableObstacleNames = nullptr;
    std::vector<string> *_staticObstacleNames = nullptr;
    bool _USE_PROBABILISTIC_UPDATE = true;
    Result _result = NullResult(0.0);
    bool _isOptimizing = false;

    // Local mujoco helpers to be used in threads when sampling.
    std::vector<shared_ptr<MujocoHelper>> _mujocoHelpers;
};

class InteractiveOptimizerBase : public OptimizerBase {
public:
    InteractiveOptimizerBase(const shared_ptr<MujocoHelper> &mujocoHelper) : OptimizerBase(mujocoHelper), _controlSequence((unsigned int) 0), _stateSequence((unsigned int) 0) {
		_autonomousCostPredictionService = _nh.serviceClient<pgf::PredictionInput>("predict_autonomous");
		_humanCostPredictionService = _nh.serviceClient<pgf::PredictionInput>("predict_human");
        _humanRequester = _nh.serviceClient<pgf::RobotBid>("bid");
    }

    void logIndividualCosts() {
        _logIndividuals = true;
    }

    void writeIndividualCostsToFile() {
        ifstream inputFile(EXPERIMENT_PATH + "next_experiment_id.txt");
        int experimentNumber;
        inputFile >> experimentNumber;

        auto individualCosts = _bestCostSequence.getIndividualCosts();
        std::ofstream outfile;
        outfile.open(EXPERIMENT_PATH + "runs.txt", std::ios_base::app);

        outfile << _iteration << ", ";
        outfile << experimentNumber << ", ";
        outfile << _actionId << ", ";
        outfile << _actionType << ", ";
        for(unsigned int i = 0; i < individualCosts.size() - 1; ++i) {
            outfile << individualCosts.at(i) << ", ";
        }

        outfile << individualCosts.at(individualCosts.size() - 1) << ", ";
        outfile << _bestCostSequence.getNumberOfObjectsInCollision() << "\n";
    }

    void predict() {
        unsigned int lookAhead = _iteration + 10;

        // Autonomous Prediction
        if (_autonomousCostPredictionService.call(_predictionMessage)) {
            float predictedCost = _predictionMessage.response.predictedCost < 0.0 ? 0.0 : _predictionMessage.response.predictedCost;
            _autonomousPredictions.push_back(Prediction(lookAhead, "PREDICTED_AUTONOMOUS", predictedCost));
            printf("Autonomous prediction iteration %d: %f\n", lookAhead, predictedCost);
        } else {
            throw std::runtime_error("Unable to do autonomous prediction.");
        }

        // Human Prediction
        if (_humanCostPredictionService.call(_predictionMessage)) {
            float predictedCost = _predictionMessage.response.predictedCost < 0.0 ? 0.0 : _predictionMessage.response.predictedCost;
            _humanPredictions.push_back(Prediction(lookAhead, "PREDICTED_HUMAN", predictedCost));
            printf("Human prediction iteration %d: %f\n", lookAhead, predictedCost);
        } else {
            throw std::runtime_error("Unable to do human prediction.");
        }
    }

    double getDecrease() {
        return _decrease;
    }

    Result optimize() override {
        if(_logIndividuals)
            _actionId++;

        if (_subGoalSet) {
            _actionType = "pushing";
            _prevIteration = _iteration;
            _iteration = 0;
        } else {
            if (_actionType == "pushing") {
                _actionType = "reaching_after_human_input";
                _iteration = _prevIteration;
            }
        }

        Result result(false, 0.0, _controlSequence);
        _result = result;

        bool secondaryConditionNotMet;
        if (_isAdaptive || _isPredictive)
            secondaryConditionNotMet = true;
        else
            secondaryConditionNotMet = _optimizationTimeInSeconds < _maxTimeOutForHelpLimit;

        int consecutiveNotReducingCost = 0;
        int consecutiveLocalMinima = 0;
        double previousCost = numeric_limits<double>::min();
		unsigned int predictionIterations = 0;

        for(int i = 0; i < 3; ++i) {
			_predictionMessage.request.distanceToGoal.push_back(0.0);
			_predictionMessage.request.highForce.push_back(0.0);
			_predictionMessage.request.blockingObstacleInHand.push_back(0.0);
        }

        _isOptimizing = true;
        auto optimizationStart = std::chrono::high_resolution_clock::now();

        _optimizationTimeInSeconds = 0.0;
        if (!_shouldReplan) {
            if (!_samplingVarianceVector) {
                throw std::runtime_error("The variance vector parameter was not set. Quitting");
            }

            // This is an initial candidate assuming to be the best so far.
            auto initialControlSequenceAndCost = getInitialControlSequenceAndItsCost();
            _controlSequence = get<0>(initialControlSequenceAndCost);
            _stateSequence = get<1>(initialControlSequenceAndCost);
            _bestCostSequence = get<2>(initialControlSequenceAndCost);
            _bestCost = _bestCostSequence.sum();



            if (_controlSequence.size() == 0) {
                printf("\033[0;31mNot able to generate a valid, initial control sequence. Optimisation aborted.\033[0m\n");
                return NullResult(0);
            }
        } else {
            printf("Warm-starting!\n");

            State finalCurrentState = _stateSequence.getState(_n - 1);
            setMuJoCoTo(finalCurrentState);

            // Current position of goal object.
            double goalObjectX = _mujocoHelper->getBodyXpos("object_3");
            double goalObjectY = _mujocoHelper->getBodyYpos("object_3");

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

            int stepsLeft = 0;
            vector<int> stepsToUpdate;
            for (unsigned int i = 0; i < _controlSequence.size(); ++i) {
                if (_controlSequence.getControl(i).isNull()) {
                    stepsLeft++;
                    stepsToUpdate.push_back(i);
                }
            }

            double linearX = eeToGoalX / stepsLeft;
            double linearY = eeToGoalY / stepsLeft;
            double angularZ = angle / stepsLeft;

            for (int i = 0; i < stepsLeft; ++i) {
                _controlSequence.getPointerToControl(stepsToUpdate[i])->setLinearX(linearX);
                _controlSequence.getPointerToControl(stepsToUpdate[i])->setLinearY(linearY);
                _controlSequence.getPointerToControl(stepsToUpdate[i])->setAngularZ(angularZ);
            }

            _mujocoHelper->resetSimulation();

            _shouldReplan = false;
        }

        bool costThresholdNotReached = _bestCost > _costThreshold;
        bool totalTimeLimitNotReached = _optimizationTimeInSeconds < _maxOptimizationTimeSeconds;

        vector<double> costSequence = {_bestCost};

        auto optimizationFinish = std::chrono::high_resolution_clock::now();
        _optimizationTimeInSeconds += formatDurationToSeconds(optimizationStart, optimizationFinish);
        optimizationStart = std::chrono::high_resolution_clock::now();
        _bid.request.name = _robotName;

        bool startPredicting = false;

        while (costThresholdNotReached && totalTimeLimitNotReached && secondaryConditionNotMet) {
        	auto iterationStart = std::chrono::high_resolution_clock::now();
            if(_logIndividuals) writeIndividualCostsToFile();

            _iteration++;

            resetToInitialState();
            auto rollouts = createNoisyTrajectories(_controlSequence);
            resetToInitialState();

            ControlSequence currentControlSequence = updateTrajectory(_controlSequence, rollouts);

            auto currentRollout = trajectoryRollout(currentControlSequence);
            auto currentCostSequence = get<1>(currentRollout);
            _currentCost = currentCostSequence.sum();

            printf("%d. Prior Cost: %f | Updated Cost: %f\n", _iteration, _bestCost, _currentCost);
            costSequence.push_back(_currentCost);

            if (_currentCost < _bestCost) {
                _controlSequence = currentControlSequence;
                _stateSequence = get<0>(currentRollout);
                _bestCostSequence = currentCostSequence;
                _bestCost = _currentCost;

                Result result2(false, 0.0, _controlSequence);
                _result = result2;
                consecutiveNotReducingCost = 0;
                consecutiveLocalMinima = 0;
            } else {
                consecutiveNotReducingCost++;
            }
			
            // Prediction
            if(_isPredictive && (_actionType == "reaching" || _actionType == "reaching_after_human_input")) {
                auto individualCosts = _bestCostSequence.getIndividualCosts();
                double distanceToGoal = individualCosts[4];
                double highForce = individualCosts[1];
                double blockingObstacleInHand = individualCosts[5];

                _predictionMessage.request.numberOfMovableObjects = _movableObstacleNames->size();
                _predictionMessage.request.numberOfCollisions = _bestCostSequence.getNumberOfObjectsInCollision();
                _predictionMessage.request.distanceToGoal[predictionIterations] = distanceToGoal;
                _predictionMessage.request.highForce[predictionIterations] = highForce;
                _predictionMessage.request.blockingObstacleInHand[predictionIterations] = blockingObstacleInHand;

                predictionIterations++;

                if(startPredicting || predictionIterations >= 3) {
                    startPredicting = true;
                    predict();
                    _predictedIterations.push_back(_iteration + 10);

                    if(predictionIterations == 3)
                        predictionIterations = 0;

                    auto autonomousPrediction = _autonomousPredictions[_autonomousPredictions.size() - 1];
                    double totalAutonomous = autonomousPrediction.getPredictedCost();

                    auto humanPrediction = _humanPredictions[_humanPredictions.size() - 1];
                    double totalHuman = humanPrediction.getPredictedCost();

                    _decrease = totalAutonomous - totalHuman;
                    if (_decrease > _thresholdPrediction) {
                        auto timeWaitingForHuman = std::chrono::high_resolution_clock::now();

                        _bid.request.predicted_gain = _decrease;
                        if (_humanRequester.call(_bid)) {
			                _humanAssigned = _bid.response.access_to_human_granted;
                            secondaryConditionNotMet = !_humanAssigned;
                        } else {
                            ROS_ERROR("Failed to request human");
                        }

                        auto timeWaitingForHumanEnd = std::chrono::high_resolution_clock::now();
                        _totalTimeWaitingForHuman += formatDurationToSeconds(timeWaitingForHuman, timeWaitingForHumanEnd);
                    }

                    printf("Human predicted %f and autonomous %f (decrease %f/%f)\n", totalHuman, totalAutonomous, _decrease, _thresholdPrediction);
                }

                if(std::find(_predictedIterations.begin(), _predictedIterations.end(), _iteration) != _predictedIterations.end()) {
                    string name = _actionType == "reaching" ? "ACTUAL_AUTONOMOUS" : "ACTUAL_AUTONOMOUS_AFTER_HUMAN_INPUT";
                    _actualCosts.push_back(Prediction(_iteration, name, distanceToGoal + highForce + blockingObstacleInHand));
                }
            }

            if(_isAdaptiveLocalMinima) {
                if (abs(previousCost - _currentCost) < 50.0 || _currentCost - _bestCost > 1000.0)
                    consecutiveLocalMinima++;
                else
                    consecutiveLocalMinima = 0;
            }

            if (_isAdaptive) {
                if (_isAdaptiveLocalMinima) {
                    if (consecutiveLocalMinima >= 2 && !_humanAssigned) {
                        auto timeWaitingForHuman = std::chrono::high_resolution_clock::now();
                        printf("Adaptively (local-minima) timed-out.\n");
                        _bid.request.predicted_gain = _isPredictive ? _decrease + 100000 : 100000;
                        if (_humanRequester.call(_bid)) {
			                _humanAssigned = _bid.response.access_to_human_granted;
                            secondaryConditionNotMet = !_humanAssigned;
                        } else {
                            ROS_ERROR("Failed to request human");
                        }
                        auto timeWaitingForHumanEnd = std::chrono::high_resolution_clock::now();
                        _totalTimeWaitingForHuman += formatDurationToSeconds(timeWaitingForHuman, timeWaitingForHumanEnd);
                    }
                } else {
                    if (consecutiveNotReducingCost >= _numberOfConsecutiveNonDecreases) {
                        printf("Adaptively timed-out.\n");
                        secondaryConditionNotMet = false;
                    }
                }
            } else if (!_isPredictive) {
                secondaryConditionNotMet = _optimizationTimeInSeconds < _maxTimeOutForHelpLimit;
            }

            if(_actionType == "pushing") { // We are pushing
                secondaryConditionNotMet = _optimizationTimeInSeconds < 30.0;
            }

            previousCost = _currentCost;

            costThresholdNotReached = _bestCost > _costThreshold;
            optimizationFinish = std::chrono::high_resolution_clock::now();
            _optimizationTimeInSeconds += formatDurationToSeconds(optimizationStart, optimizationFinish);
            totalTimeLimitNotReached = _optimizationTimeInSeconds < _maxOptimizationTimeSeconds;
            optimizationStart = std::chrono::high_resolution_clock::now();
            auto iterationFinish = std::chrono::high_resolution_clock::now();
            auto diff = formatDurationToSeconds(iterationStart, iterationFinish);
	        printf("Iteration time: %f\n", diff);
        }

        resetToInitialState();

        auto bestStateSequence = get<0>(trajectoryRollout(_controlSequence));
        State finalState = bestStateSequence.getState(bestStateSequence.size() - 1);
        _tempState = finalState;

        optimizationFinish = std::chrono::high_resolution_clock::now();
        _optimizationTimeInSeconds += formatDurationToSeconds(optimizationStart, optimizationFinish);
        printf("Optimization time: %f\n", _optimizationTimeInSeconds);

        bool reachedGoal = false;
        if (_bestCost <= _costThreshold) {
            if (_subGoalSet) { // We were optimizing for a sub-goal problem.
                reachedGoal = isSubGoalAchieved(finalState);
                _reachedSubGoal = reachedGoal;
                if (reachedGoal) {
                    _subGoalSet = false;
                }
            } else {
                reachedGoal = isPrimaryGoalAchieved(finalState);
                _reachedPrimaryGoal = reachedGoal;
            }
        }

        if(_logIndividuals)
            writeIndividualCostsToFile();

        _logIndividuals = false;

        if(_actionType == "pushing" && reachedGoal) {
            logHumanInput();
        }

        Result finalResult(reachedGoal, _optimizationTimeInSeconds, _controlSequence);
        finalResult.setCostSequence(costSequence);
        _result = finalResult;
        _isOptimizing = false;
        return finalResult;
    }

    void logHumanInput() {
        ifstream inputFile(EXPERIMENT_PATH + "next_experiment_id.txt");
        int experimentNumber;
        inputFile >> experimentNumber;

        std::ofstream outfile;
        outfile.open(EXPERIMENT_PATH + "human_inputs.txt", std::ios_base::app);

        outfile << experimentNumber << ", ";
        outfile << _prevIteration << ", ";
        outfile << _goalObjectName << ", ";
        outfile << _subGoalPrevX << ", ";
        outfile << _subGoalPrevY << ", ";
        outfile << _subGoalDesiredX << ", ";
        outfile << _subGoalDesiredY << "\n";
    }

    // MPPI algorithm to follow the optimized _controlSequence trajectory already optimized from this.optimize().
    Control followAlreadyOptimizedTrajectory(string actionType) {
        _mujocoHelper->resetSimulation();

        if (!_samplingVarianceVector) {
            throw std::runtime_error("The variance vector parameter was not set. Quiting");
        }

        if (_controlSequence.size() == 0) {
            // This is an initial candidate assuming to be the best so far.
            auto initialControlSequenceAndCost = getInitialControlSequenceAndItsCost();
            _controlSequence = get<0>(initialControlSequenceAndCost);
            _stateSequence = get<1>(initialControlSequenceAndCost);
            _bestCostSequence = get<2>(initialControlSequenceAndCost);
            _bestCost = _bestCostSequence.sum();

            if (_controlSequence.size() == 0) {
                printf("\033[0;31mNot able to generate a valid, initial control sequence. Optimisation aborted.\033[0m\n");
                return {0.0, 0.0, 0.0, 0.0};
            }
        }

        State currentState = _initialState;
        _tempState = currentState;

        if (actionType == "push") { // We were optimizing for a sub-goal problem.
            _reachedSubGoal = isSubGoalAchieved(currentState);
            if (_reachedSubGoal) {
                printf("Sub-goal achieved in followAlreadyOptimizedTrajectory, sending zero vel!\n");
                return {0.0, 0.0, 0.0, 0.0};
            }
        } else {
            _reachedPrimaryGoal = isPrimaryGoalAchieved(currentState);
            if (_reachedPrimaryGoal) {
                return {0.0, 0.0, 0.0, 0.0};
            }
        }

        auto currentRollout = trajectoryRollout(_controlSequence);
        _bestCostSequence = get<1>(currentRollout);
        _bestCost = _bestCostSequence.sum();

        // Check if we can reach the goal state
        auto stateSequence = get<0>(currentRollout);
        State finalState = stateSequence.getState(stateSequence.size() - 1);

        bool reachedGoal;
        if (actionType == "push") { // We were optimizing for a sub-goal problem.
            reachedGoal = isSubGoalAchieved(finalState);
            _reachedSubGoal = reachedGoal;
        } else {
            reachedGoal = isPrimaryGoalAchieved(finalState);
            _reachedPrimaryGoal = reachedGoal;
        }

        if (reachedGoal) {
            auto firstControl = _controlSequence.getControl(0);
            _controlSequence = removeFirstStepAndExpandTrajectoryByOneStep(_controlSequence);
            return firstControl;
        } else {
            printf("Sorry, you should replan! The cost is: %f\n", _bestCost);
            if (actionType == "push")
                _subGoalSet = true;
            _shouldReplan = true;
        }

        return {0, 0, 0, 0};
    }

    State getTempState() {
        return _tempState;
    }

    bool isHumanAssigned() {
       return _humanAssigned;
    }

    void unAssignHuman() {
       _humanAssigned = false;
    }

    void updateIfWeReachedTheGoal(string actionType) {
        State currentState = _mujocoHelper->getLatestSavedState();

        if (actionType == "push") { // We were optimizing for a sub-goal problem.
            _reachedSubGoal = isSubGoalAchieved(currentState);
            if (!_reachedSubGoal) {
                _subGoalSet = true;
            }
        } else {
            _reachedPrimaryGoal = isPrimaryGoalAchieved(currentState);
        }
    }

    void clearSubGoal() {
        _subGoalSet = false;
    }

    // Getters

    bool shouldReplan() {
        return _shouldReplan;
    }

    bool reachedPrimaryGoal() {
        return _reachedPrimaryGoal;
    }

    bool reachedSubGoal() {
        return _reachedSubGoal;
    }

    ControlSequence getControlSequence() {
        return _controlSequence;
    }

    double getCurrentBestTotalCost() {
        return _bestCostSequence.sum();
    }

    // Setters
    void setThresholdPrediction(float threshold) {
        _thresholdPrediction = threshold;
    }

    void setMaxOptimizationTime(double maxTimeInSeconds) {
        _maxOptimizationTimeSeconds = maxTimeInSeconds;
    }

    void setSubGoal(double x, double y) {
        _subGoalPrevX = _mujocoHelper->getBodyXpos(_goalObjectName);
        _subGoalPrevY = _mujocoHelper->getBodyYpos(_goalObjectName);
        _subGoalDesiredX = x;
        _subGoalDesiredY = y;
        _reachedSubGoal = false;
        _reachedPrimaryGoal = false;
        _subGoalSet = true;
    }

    void updateMujocoHelpers(const shared_ptr<MujocoHelper> &mujocoHelper) {
        // Create a MuJoCo Helper for each of the K noisy trajectories (for palatalization).
        createLocalMujocoHelpers(mujocoHelper);
        _mujocoHelper = make_shared<MujocoHelper>(mujocoHelper.get());
    }

    void setAdaptiveLocalMinima(bool isAdaptiveLocalMinima) {
        _isAdaptiveLocalMinima = isAdaptiveLocalMinima;
    }

    void setAdaptive(bool isAdaptive) {
        _isAdaptive = isAdaptive;
    }

    void setTimeOutForHelpLimit(double timeLimit) {
        _maxTimeOutForHelpLimit = timeLimit;
    }

    void setAdaptiveNumberOfConesecutiveNonDecreases(int number) {
        _numberOfConsecutiveNonDecreases = number;
    }

    void enablePredictiveTimeout(bool isPredictive) {
        _isPredictive = isPredictive;
    }

    vector<Prediction> getActualCosts() {
        return _actualCosts;
    }

    vector<Prediction> getAutonomousPredictions() {
        return _autonomousPredictions;
    }

    vector<Prediction> getHumanPredictions() {
        return _humanPredictions;
    }

    double getTotalTimeWaitingForHuman() {
        return _totalTimeWaitingForHuman;
    }

    double getTotalTimeWastedWaitingForHuman() {
        return _totalTimeWastedWaitingForHuman;
    }

    void setRobotName(string name) {
        _robotName = name;
    }
protected:
    virtual bool isPrimaryGoalAchieved(State) = 0;

    virtual bool isSubGoalAchieved(State) = 0;

    ControlSequence removeFirstStepAndExpandTrajectoryByOneStep(ControlSequence controlSequence) {
        ControlSequence newControlSequence(_n);
        for (int i = 1; i < _n; ++i) {
            Control currentControl = controlSequence.getControl(i);
            newControlSequence.addControl(currentControl);
        }

        Control velocitiesWithOpenGripper(0.0, 0.0, 0.0, 0.0);
        newControlSequence.addControl(velocitiesWithOpenGripper);
        // newControlSequence.addControl(controlSequence.getControl(_n - 1));
        return newControlSequence;
    }

	ros::NodeHandle _nh;
	ros::ServiceClient _autonomousCostPredictionService;
	ros::ServiceClient _humanCostPredictionService;
    ros::ServiceClient _humanRequester;
    pgf::RobotBid _bid;
	pgf::PredictionInput _predictionMessage;
    vector<Prediction> _autonomousPredictions = {};
    vector<Prediction> _humanPredictions = {};
    vector<Prediction> _actualCosts = {};
    vector<unsigned int> _predictedIterations = {};
    double _currentCost;
    string _robotName;

    float _thresholdPrediction = 0.01;

    double _decrease = 0.0;
    double _maxTimeOutForHelpLimit = 0.0;
    int _numberOfConsecutiveNonDecreases;
    ControlSequence _controlSequence;
    StateSequence _stateSequence;
    double _bestCost = std::numeric_limits<double>::max();
    CostSequence _bestCostSequence;
    bool _shouldReplan = false;
    bool _reachedPrimaryGoal = false;
    bool _reachedSubGoal = false;
    bool _subGoalSet = false;
    bool _isAdaptive = false;
    bool _isAdaptiveLocalMinima = false;
    bool _isPredictive = false;
    double _subGoalDesiredX;
    double _subGoalDesiredY;
    double _subGoalPrevX;
    double _subGoalPrevY;
    double _maxOptimizationTimeSeconds = std::numeric_limits<double>::max();
    State _tempState;
    double _optimizationTimeInSeconds = 0.0;
    bool _logIndividuals = false;
    int _actionId = 0;
    string _actionType = "reaching";
    int _iteration = 0;
    int _prevIteration = 0;
    bool _print = false;
    string _goalObjectName = "object_3";
    double _totalTimeWastedWaitingForHuman = 0.0;
    double _totalTimeWaitingForHuman = 0.0;
    bool _humanAssigned = false;
};

#endif
