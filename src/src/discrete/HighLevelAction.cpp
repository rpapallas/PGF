class HighLevelAction {
public:
    HighLevelAction(string objectName, double goalX, double goalY) {
        _objectName = objectName;
        _goalX = goalX;
        _goalY = goalY;
    }

    string getName() {
        return _objectName;
    }

    double getX() {
        return _goalX;
    }

    double getY() {
        return _goalY;
    }

    bool operator==(HighLevelAction &rhs) {
        return getName() == rhs.getName();
    }

private:
    string _objectName;
    double _goalX;
    double _goalY;
};


class Guidance {
public:
    Guidance() = default;

    void add(const HighLevelAction& highLevelAction) {
        _highLevelActions.push_back(highLevelAction);
    }

    vector <HighLevelAction> getHighLevelActions() {
        return _highLevelActions;
    }

    HighLevelAction getHighLevelAction(int i) {
        return _highLevelActions[i];
    }

    unsigned int size() {
        return _highLevelActions.size();
    }

private:
    vector <HighLevelAction> _highLevelActions;
};
