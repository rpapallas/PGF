cmake_minimum_required (VERSION 2.8.3)
project(pgf)

set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(compilerSettings)
include(boost)
include(includes)
include(mujoco)
include(openmp)

include_directories(${Boost_INCLUDE_DIRS} 
                    ${OMPL_INCLUDE_DIRS} 
                    ${catkin_INCLUDE_DIRS}
                    ${Eigen_INCLUDE_DIRS} 
                    ${Mujoco_INCLUDE_DIRS}
                    "include")

link_directories(${Boost_LIBRARY_DIRS}
                 ${OMPL_LIBRARY_DIRS}
                 ${CMAKE_CURRENT_SOURCE_DIR}/lib/mjpro200/)



add_executable(pgf src/demos/PGF.cpp)
target_link_libraries(pgf
        ${catkin_LIBRARIES}
        ${OMPL_LIBRARIES}
        ${LIB_MUJOCO}
        ${GLFW}
        libGL.so
        libglew.so
        ${CMAKE_THREAD_LIBS_INIT}  # To have -pthread
        GL
        GLU
        glut
        -fopenmp
        -lpthread
        -lboost_thread
        -lboost_system)
