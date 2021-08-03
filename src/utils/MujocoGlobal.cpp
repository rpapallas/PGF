#include "mujoco.h"
#include <GL/glew.h>
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <cstdio>
#include <chrono>
#include <thread>
#include <fstream>
#include <random>
#include "../utils/MujocoHelper.cpp"
#include "../utils/MujocoUI.cpp"
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

// MuJoCo data structures
mjModel *model = NULL;                  // MuJoCo model
mjData *data = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
mjvPerturb pert;
GLFWwindow *window;
shared_ptr<MujocoHelper> globalMujocoHelper;

double SIMULATION_TIMESTEP = 0.0015;

bool LIVE_ENV_TRACKING = false;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
bool NOESC = false;
double lastx = 0;
double lasty = 0;
bool userWantsToInteract = false;

int WINDOW_WIDTH = 1200;
int WINDOW_HEIGHT = 900;

bool isMouseLocked = false;
bool isKeyboardLocked = false;
bool isZoomLocked = false;

bool objectSelected = false;
double clickedX, clickedY;
bool finishedExecution = false;
const string PROJECT_ROOT_PATH = "/home/rafael/catkin_ws/src/predictive-guided-framework/";
const string SCENE_PATH = PROJECT_ROOT_PATH + "models/";
const string MJ_KEY_PATH = "/home/rafael/catkin_ws/mjkey.txt";
string EXPERIMENT_PATH = "";
