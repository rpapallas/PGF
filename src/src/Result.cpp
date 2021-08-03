#include <utility>

class Result {
public:
    Result(bool isSuccessful, double optimizationTime, ControlSequence controlSequence) {
        _isSuccessful = isSuccessful;
        _optimizationTime = optimizationTime;
        _controlSequence = std::move(controlSequence);
    }

    void setCostSequence(vector<double> costSequence) {
        _costSequence = costSequence;
    }

    bool isSuccessful() {
        return _isSuccessful;
    }

    double getOptimizationTime() {
        return _optimizationTime;
    }

    ControlSequence getControlSequence() {
        return _controlSequence;
    }

    vector<double> getCostSequence() {
        return _costSequence;
    }

private:
    bool _isSuccessful;
    double _optimizationTime;
    ControlSequence _controlSequence;
    vector<double> _costSequence;
};


class NullResult : public Result {
public:
    explicit NullResult(double optimizationTime) : Result(false, optimizationTime, ControlSequence((unsigned int) 0)) {
    }
};