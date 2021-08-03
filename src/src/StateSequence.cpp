class StateSequence {
public:
    StateSequence(int n) : _steps(n) {
    }

    void addState(State state) {
        if (_states.size() < _steps) {
            _states.push_back(state);
        } else {
            throw std::invalid_argument(
                    "You are attempting to add more states to a control sequence than the maximum.");
        }
    }

    vector<State> getStates() {
        if (_states.size() < _steps) {
            printf("NOTE: You are trying to access the states of a ControlSequence that currently contains less controls than the allowed number of controls, be careful.\n");
        }

        return _states;
    }

    State getState(unsigned int index) {
        if (index >= _states.size()) {
            throw std::invalid_argument("You are attempting to access a state index that is out of bounds.");
        }

        return _states[index];
    }

    unsigned int size() {
        return _states.size();
    }

    bool isEmpty() {
        return _states.size() == 0;
    }

private:
    unsigned int _steps;
    vector<State> _states;
};
