class CostSequence {
public:
    CostSequence() : _steps(0) {
    }

    explicit CostSequence(unsigned int steps) : _steps(steps) {
    }

    unsigned int size() {
        return _costs.size();
    }

    vector<double> getCosts() {
        if (_costs.size() < _steps) {
            printf("NOTE: You are trying to access the costs of a CostSequence that currently contains less costs than the allowed number of costs, be careful.\n");
        }

        return _costs;
    }

    double getCost(unsigned int index) {
        return _costs[index];
    }

    void addCost(double cost) {
        if (_costs.size() < _steps) {
            _costs.push_back(cost);
        } else {
            throw std::invalid_argument(
                    "You are attempting to add more controls to a control sequence than the maximum.");
        }
    }

    void addIndividualCost(double cost, int index) {
        int size = (int) _individualCosts.size();
        if(size - 1 < index) {
            _individualCosts.push_back(0.0);
        }

        _individualCosts[index] += cost;
    }

	void setNumberOfObjectsInCollision(unsigned int number) {
		_numberOfObjectsInCollision = number;
	}

	unsigned int getNumberOfObjectsInCollision() {
		return _numberOfObjectsInCollision;
	}

    vector<double> getIndividualCosts() {
        return _individualCosts;
    }

    double sum() {
        double sum = 0.0;
        for (const double cost : _costs) {
            sum += cost;
        }

        return sum;
    }

    double get(unsigned int index) {
        if (index >= _costs.size()) {
            throw std::invalid_argument("You are attempting to access a cost index that is out of bounds.");
        }

        return _costs[index];
    }

    bool isEmpty() {
        return _costs.size() == 0;
    }

    double max() {
        return *max_element(_costs.begin(), _costs.end());
    }

    double min() {
        return *min_element(_costs.begin(), _costs.end());
    }

private:
	unsigned int _numberOfObjectsInCollision = 0;
    unsigned int _steps;
    vector<double> _costs = {};
    vector<double> _individualCosts = {};
};
