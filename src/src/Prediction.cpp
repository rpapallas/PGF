class Prediction {
public:
    Prediction(unsigned int iteration, string name, double predictedCost) {
        _iteration = iteration;
        _name = name;
        _predictedCost = predictedCost;
    }

    unsigned int getIteration() {
        return _iteration;
    }

    double getPredictedCost() {
        return _predictedCost;
    }

    string getName() {
        return _name;
    }
private:
    unsigned int _iteration;
    string _name;
    double _predictedCost;
};

