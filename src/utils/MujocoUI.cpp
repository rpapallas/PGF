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

#include <boost/algorithm/string.hpp>

extern bool LIVE_ENV_TRACKING;

// MuJoCo data structures
extern mjModel *model;                  // MuJoCo model
extern mjData *data;                   // MuJoCo data
extern mjvCamera cam;                      // abstract camera
extern mjvOption opt;                      // visualization options
extern mjvScene scn;                       // abstract scene
extern mjrContext con;                     // custom GPU context
extern mjvPerturb pert;
extern GLFWwindow *window;
extern shared_ptr<MujocoHelper> globalMujocoHelper;
extern double SIMULATION_TIMESTEP;

// mouse interaction
extern bool button_left;
extern bool button_middle;
extern bool button_right;
extern bool NOESC;
extern double lastx;
extern double lasty;
extern bool userWantsToInteract;

extern int WINDOW_WIDTH;
extern int WINDOW_HEIGHT;

extern bool isMouseLocked;
extern bool isKeyboardLocked;
extern bool isZoomLocked;

extern bool objectSelected;
extern double clickedX, clickedY;

void render() {
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(model, data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
}

// keyboard callback
void verbose() {
    printf("Ridgeback Position | x: %f, y: %f, yaw: %f\n", globalMujocoHelper->getRobotXpos(),
           globalMujocoHelper->getRobotYpos(), globalMujocoHelper->getRobotYaw());

    Eigen::MatrixXf handTransform = globalMujocoHelper->getEndEffectorTransform();
    const double ee_x = handTransform(0, 3);
    const double ee_y = handTransform(1, 3);
    boost::array<double, 4> eeQuat = globalMujocoHelper->getQuatFromTransform(handTransform);
    double theta = globalMujocoHelper->getYawFromQuat(eeQuat[0], eeQuat[1], eeQuat[2], eeQuat[3]);
    printf("End-Effector Position: | x: %f, y: %f, theta: %f\n", ee_x, ee_y, theta);

    if (globalMujocoHelper->isHighForceAppliedToShelf())
        printf("High force applied to shelf.\n");


    vector<mjtNum *> bodyForceTorques = globalMujocoHelper->getBodyForceTorques("shelf");
    printf("Forces: ");
    for (mjtNum *forceTorque : bodyForceTorques) {
        for (int i = 0; i < 3; ++i) { // 3 forces
            printf("%f ", forceTorque[i]);
        }
    }
    printf("\n");

    double dist = globalMujocoHelper->getDistanceOfEndEffectorTo("object_3");
    printf("Distance of EE to Goal Object (Green): %f\n", dist);

    printf("\nObject velocities:\n");
    for(int i = 1; i <= 20; ++i) {
        string objectName = "object_" + to_string(i);
        auto objectVelocities = globalMujocoHelper->getBodyVelocity(objectName);
        double v1 = get<0>(objectVelocities);
        double v2 = get<1>(objectVelocities);
        double v6 = get<5>(objectVelocities);
        printf("%s: %f, %f, %f\n", objectName.c_str(), v1, v2, v6);
    }

    printf("\nDistance end-effector velocities:\n");
    for(int i = 1; i <= 20; ++i) {
        string objectName = "object_" + to_string(i);
        double dist = globalMujocoHelper->getDistanceOfEndEffectorTo(objectName);

        vector<mjtNum *> bodyForceTorques = globalMujocoHelper->getBodyForceTorques(objectName);
        for (mjtNum *forceTorque : bodyForceTorques) {
            double forceMagnitude = sqrt(forceTorque[0] * forceTorque[0] + forceTorque[1] * forceTorque[1] + forceTorque[2] * forceTorque[2]);
            printf("%s | dist: %f | force: %f\n", objectName.c_str(), dist, forceMagnitude);
        }
    }

    printf("Camera settings\n");
    printf("cam.lookat[0]: %f\n", cam.lookat[0]);
    printf("cam.lookat[1]: %f\n", cam.lookat[1]);
    printf("cam.lookat[2]: %f\n", cam.lookat[2]);
    printf("cam.distance: %f\n", cam.distance);
    printf("cam.azimuth: %f\n", cam.azimuth);
    printf("cam.elevation: %f\n", cam.elevation);

    printf("\n");
}


void keyboard(GLFWwindow * /* window */, int key, int /* scancode */, int act, int /* mods */) {
    if (isKeyboardLocked)
        return;

    const bool isVerbose = true;

    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        globalMujocoHelper->resetSimulation();
    }

    if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        NOESC = true;
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_I) {
        Eigen::MatrixXf endEffectorTransform = globalMujocoHelper->getEndEffectorTransform();
        auto solutions = globalMujocoHelper->ik(endEffectorTransform);
        for(const auto& solution : solutions) {
            globalMujocoHelper->setUR5JointValues(solution);
            globalMujocoHelper->forward();
            render();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_K) {
        globalMujocoHelper->disableCollisionsFor("object_1");
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_L) {
        globalMujocoHelper->enableCollisionsFor("object_1");
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_C) { // Close Gripper
        globalMujocoHelper->setGripperDOFValue(250);
        globalMujocoHelper->setRobotVelocity(0.0, 0.0, 0.0);
        globalMujocoHelper->step();
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_O) { // Open Gripper
        globalMujocoHelper->setGripperDOFValue(0);
        globalMujocoHelper->setRobotVelocity(0.0, 0.0, 0.0);
        globalMujocoHelper->step();
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_W) { // Forward
        for (int i = 0; i < 50; ++i) {
            globalMujocoHelper->setRobotVelocity(0.1, 0.0, 0.0);
            globalMujocoHelper->step();
        }
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_S) { // Back
        for (int i = 0; i < 50; ++i) {
            globalMujocoHelper->setRobotVelocity(-0.1, 0.0, 0.0);
            globalMujocoHelper->step();
        }
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_D) { // Right
        for (int i = 0; i < 50; ++i) {
            globalMujocoHelper->setRobotVelocity(0.0, -0.1, 0.0);
            globalMujocoHelper->step();
        }
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_A) { // Left
        for (int i = 0; i < 50; ++i) {
            globalMujocoHelper->setRobotVelocity(0.0, 0.1, 0.0);
            globalMujocoHelper->step();
        }
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_E) { // Rotate clockwise
        for (int i = 0; i < 50; ++i) {
            globalMujocoHelper->setRobotVelocity(0.0, 0.0, 0.1);
            globalMujocoHelper->step();
        }
    }

    if ((act == GLFW_PRESS || act == GLFW_REPEAT) && key == GLFW_KEY_Q) { // Rotate anti-clockwise
        for (int i = 0; i < 50; ++i) {
            globalMujocoHelper->setRobotVelocity(0.0, 0.0, -0.1);
            globalMujocoHelper->step();
        }

    }

    if (isVerbose && act == GLFW_PRESS && key == GLFW_KEY_V) {
        verbose();
    }

}

void tideUp() {
    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(data);
    mj_deleteModel(model);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
}

void windowCloseCallback(GLFWwindow * /*window*/) {
    // Use this flag if you wish not to terminate now.
    // glfwSetWindowShouldClose(window, GLFW_FALSE);
    tideUp();
}

std::tuple<GLdouble, GLdouble> ScreenToWorld(int x, int y) {
    GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];
    GLfloat winX, winY, winZ;
    GLdouble posX, posY, posZ;

    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetIntegerv(GL_VIEWPORT, viewport);

    winX = (float) x;
    winY = (float) viewport[3] - (float) y;
    glReadPixels(x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);

    gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

    return std::make_tuple(posX, posY);
}

// mouse button callback
void mouse_button(GLFWwindow *window, int /* button */, int /* act */, int /* mods */) {
    if (isMouseLocked)
        return;

    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);

    if (button_middle) {
        userWantsToInteract = true;
        std::tuple<GLdouble, GLdouble> selectionCoordinate = ScreenToWorld(lastx, lasty);
        clickedX = std::get<0>(selectionCoordinate);
        clickedY = std::get<1>(selectionCoordinate);
        objectSelected = true;
    }
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
    if (isMouseLocked)
        return;

    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(model, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow * /* window */, double /* xoffset */, double yoffset) {
    if (isMouseLocked)
        return;

    if (!isZoomLocked) {
        // emulate vertical mouse motion = 5% of window height
        mjv_moveCamera(model, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
    }
}

void mujocoConfiguration(const std::string &scenePath, int parallelWindowIndex) {
    // load and compile model
    char error[1000] = "Could not load binary model";
    model = mj_loadXML(scenePath.c_str(), 0, error, 1000);

    // make data
    data = mj_makeData(model);

    // Simulation step size. Change this one if you want non-real-time simulation.
    model->opt.timestep = SIMULATION_TIMESTEP;

    // init GLFW
    if (!glfwInit())
        mju_error("Could not initialize GLFW");

    WINDOW_HEIGHT = 500;
    WINDOW_WIDTH = 620;

    int ypos = 20;
    if(parallelWindowIndex > 3)
        ypos = WINDOW_HEIGHT + 350;

    int xpos = 50;
    if (parallelWindowIndex > 1) {
        if(parallelWindowIndex > 3) {
            parallelWindowIndex = parallelWindowIndex - 3;
        }
        xpos = WINDOW_WIDTH;
        xpos = 55 + (xpos * (parallelWindowIndex - 1));
    }

    // create window, make OpenGL context current, request v-sync
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(WINDOW_WIDTH,
                              WINDOW_HEIGHT,
                              "",
                              NULL,
                              NULL);

    glfwSetWindowPos(window, xpos, ypos);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(model, &scn, 2000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetWindowCloseCallback(window, windowCloseCallback);

    // Hide the window for now.
    glfwHideWindow(window);

}

void loadModel(const std::string &scenePath) {
    mjModel *mnew = 0;
    char error[500] = "";
    mnew = mj_loadXML(scenePath.c_str(), NULL, error, 500);
    mj_deleteData(data);
    mj_deleteModel(model);
    model = mnew;
    data = mj_makeData(model);
    mj_forward(model, data);

    // Simulation step size. Change this one if you want non-real-time simulation.
    model->opt.timestep = SIMULATION_TIMESTEP;

    // re-create scene and context
    mjv_makeScene(model, &scn, 2000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);

    cam.lookat[0] = 1.48821;
    cam.lookat[1] = 0.00114961;
    cam.lookat[2] = 0.220028;
    cam.distance = 1.24173;
    cam.azimuth = 90.6;
    cam.elevation = -81.4;
}

string getObjectSelection(vector<string> movableObjectNames) {
    objectSelected = false;
    std::string objectNameSelected;
    bool validObjectSelected = false;

    do {
        cout << "Select an object by clicking on it." << endl;
        while (!objectSelected) {
            render();
        }
        objectSelected = false;

        for (std::string objectName : movableObjectNames) {
            double objectX = globalMujocoHelper->getBodyXpos(objectName);
            double objectY = globalMujocoHelper->getBodyYpos(objectName);

            double dx = abs(clickedX - objectX);
            double dy = abs(clickedY - objectY);

            if (dx < 0.04 && dy < 0.04) {
                validObjectSelected = true;
                objectNameSelected = objectName;
            }
        }
    } while (!validObjectSelected);

    return objectNameSelected;
}

string getObjectSelectionWithoutRendering(vector<string> movableObjectNames) {
    objectSelected = false;
    std::string objectNameSelected;
    bool validObjectSelected = false;

    do {
        cout << "Select an object by clicking on it." << endl;
        while (!objectSelected) {
        }
        objectSelected = false;

        for (std::string objectName : movableObjectNames) {
            double objectX = globalMujocoHelper->getBodyXpos(objectName);
            double objectY = globalMujocoHelper->getBodyYpos(objectName);

            double dx = abs(clickedX - objectX);
            double dy = abs(clickedY - objectY);

            if (dx < 0.04 && dy < 0.04) {
                validObjectSelected = true;
                objectNameSelected = objectName;
            }
        }
    } while (!validObjectSelected);

    return objectNameSelected;
}

tuple<double, double> getPushPosition() {
    double x = clickedX, y = clickedY;
    objectSelected = false;
    cout << "Select a position to push the object to." << endl;
    while (!objectSelected || abs(clickedX - x) <= 0.0001 || abs(clickedY - y) <= 0.0001) {
        render();
    }

    objectSelected = false;

    double positionToPushObjectX = clickedX;
    double positionToPushObjectY = clickedY;

    return make_tuple(positionToPushObjectX, positionToPushObjectY);
}

tuple<double, double> getObjectPositionWithoutRendering() {
    double x = clickedX, y = clickedY;
    objectSelected = false;
    cout << "Select a position to push the object to." << endl;

    while (!objectSelected || abs(clickedX - x) <= 0.0001 || abs(clickedY - y) <= 0.0001) {
    }

    objectSelected = false;

    double positionToPushObjectX = clickedX;
    double positionToPushObjectY = clickedY;

    return make_tuple(positionToPushObjectX, positionToPushObjectY);
}

void updateCameraSettingsFromModel() {
    // Camera settings
    double *cameraSettings = globalMujocoHelper->getNumericField("camSettings");

    if (cameraSettings == 0)
        throw std::invalid_argument("You need to set a custom attribute, camSettings, for the camera settings.");

    cam.lookat[0] = cameraSettings[0];
    cam.lookat[1] = cameraSettings[1];
    cam.lookat[2] = cameraSettings[2];
    cam.distance = cameraSettings[3];
    cam.azimuth = cameraSettings[4];
    cam.elevation = cameraSettings[5];
}


void initMujocoWithCustomWindowFrom(string sceneFileName, int parallelWindowIndex) {
    mujocoConfiguration(sceneFileName, parallelWindowIndex);
    mj_step(model, data);
    globalMujocoHelper = make_shared<MujocoHelper>(model, data, "robot", "robot_lin_x", "robot_lin_y", "robot_ang_z", 0.0, 0.0);
    updateCameraSettingsFromModel();
}

void initMujocoFrom(std::string sceneFileName) {
    initMujocoWithCustomWindowFrom(sceneFileName, 1);
}
