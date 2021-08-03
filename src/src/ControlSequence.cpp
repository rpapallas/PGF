class ControlSequence {
public:
    explicit ControlSequence(unsigned int steps) : _steps(steps) {
    }

    ControlSequence() : _steps(0) {
    }

    explicit ControlSequence(const ControlSequence* originalControlSequence) {
        _controls = originalControlSequence->getControls();
        _steps = originalControlSequence->getMaximumNumberOfSteps();
    }

    ControlSequence& operator = (const ControlSequence &other) {
        _controls = other.getControls();
        _steps = other.getMaximumNumberOfSteps();
        return *this;
    }

    int getMaximumNumberOfSteps() const {
        return _steps;
    }

    void addControl(Control control) {
        if (_controls.size() < _steps) {
            _controls.push_back(control);
        } else {
            throw std::invalid_argument(
                    "You are attempting to add more controls to a control sequence than the maximum.");

        }
    }

    vector<Control> getControls() const {
        if (_controls.size() < _steps) {
            printf("NOTE: You are trying to access the controls of a ControlSequence that currently contains less controls than the allowed number of controls, be careful.\n");
        }

        return _controls;
    }

    bool isEmpty() {
        return _controls.empty();
    }

    unsigned int size() {
        return _controls.size();
    }

    Control getControl(unsigned int index) {
        if (index >= _controls.size()) {
            throw std::invalid_argument("You are attempting to access a control index that is out of bounds.");
        }

        return _controls[index];
    }

    Control* getPointerToControl(unsigned int index) {
        if (index >= _controls.size()) {
            throw std::invalid_argument("You are attempting to access a control index that is out of bounds.");
        }

        return &_controls[index];
    }

private:
    vector<Control> _controls = {};
    unsigned int _steps;
};
