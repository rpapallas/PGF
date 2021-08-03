#ifndef COMMONFUNCS
#define COMMONFUNCS

#include "../src/discrete/MoveInFreeSpaceOptimizer.cpp"
#include <sensor_msgs/JointState.h>

void savePredictionData(int experimentId, vector<Prediction> actualValues, vector<Prediction> autonomousPredictions, vector<Prediction> humanPredictions) {
    if(autonomousPredictions.size() == 0) {
        return;
    }

    std::ofstream outfile;
    outfile.open(EXPERIMENT_PATH + "predictions.txt", std::ios_base::app);

    for(unsigned int i = 0; i < autonomousPredictions.size(); ++i) {
        Prediction autonomousPrediction = autonomousPredictions[i];
        double autonomousPredictedCost = autonomousPrediction.getPredictedCost();
        outfile << experimentId << ", " << autonomousPrediction.getIteration() << ", " << autonomousPrediction.getName() <<  ", " << autonomousPredictedCost << "\n";

        Prediction humanPrediction = humanPredictions[i];
        double humanPredictedCost = humanPrediction.getPredictedCost();
        outfile << experimentId << ", " << humanPrediction.getIteration() << ", " << humanPrediction.getName() <<  ", " << humanPredictedCost << "\n";

        unsigned int actualIteration = humanPrediction.getIteration();
        string actualName = "ACTUAL_AUTONOMOUS";
        double actualCost = 0.0;

        if(actualValues.size() > i) {
            Prediction actualValue = actualValues[i];
            actualName = actualValue.getName();
            actualIteration = actualValue.getIteration();
            actualCost = actualValue.getPredictedCost();
        } else if (actualValues.size() > 0) {
            Prediction actualValue = actualValues[actualValues.size() - 1];
            actualName = actualValue.getName();
            actualIteration = humanPrediction.getIteration(); // Just match iteration to human's prediction
            actualCost = actualValue.getPredictedCost();
        }

        outfile << experimentId << ", " << actualIteration << ", " << actualName <<  ", " << actualCost << "\n";

        if(actualCost < 20.0)
            break;
    }

    outfile << "\n";
}

void saveScreenshotToFile(const std::string &filename, int windowWidth, int windowHeight) {
    const int numberOfPixels = windowWidth * windowHeight * 3;
    unsigned char pixels[numberOfPixels];

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels);

    FILE *outputFile = fopen(filename.c_str(), "w");
    short header[] = {0, 2, 0, 0, 0, 0, (short) windowWidth, (short) windowHeight, 24};

    fwrite(&header, sizeof(header), 1, outputFile);
    fwrite(pixels, numberOfPixels, 1, outputFile);
    fclose(outputFile);
}

vector<string> getObjectNamesFromNumberOfObjects(int numberOfObjects) {
    vector<string> objectNames;
    for (int i = 0; i < numberOfObjects; ++i) {
        string objectName = "object_" + std::to_string(i + 1);
        objectNames.push_back(objectName);
    }

    return objectNames;
}

#endif
