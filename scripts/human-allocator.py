#!/usr/bin/python
import sys
import os
import rospy
import time
from os import path
from pgf.srv import HumanRelease, HumanReleaseResponse
from pgf.srv import RobotBid, RobotBidResponse
from datetime import datetime, timedelta
import multiprocessing

window_size = 2.0    # seconds
human_busy = False
bidding_robots = []
accept_new_robots = True
winner = None
lock = multiprocessing.Lock()
log_path = os.path.expanduser('~/log.txt')

def output(string):
    with lock:
        with open(log_path, 'a') as f:
            if string != '\n':
                print >> f, ('%s (%s)' % (string, get_time_str()))
            else:
                print >> f, string

def get_time_str():
    return datetime.now().strftime("%H:%M:%S.%f")

def human_release(robot):
    output('Robot %s returned human...' % robot.name)
    global human_busy
    human_busy = False
    return HumanReleaseResponse(True)

def bid_for_human(this_robot):
    if human_busy: 
        output('Robot %s with predicted gain %f tried to join the window, but human busy' % (this_robot.name, this_robot.predicted_gain))
        return RobotBidResponse(False)

    if not accept_new_robots:
        output("Robot %s with predicted gain %f tried to join the window but the window is not yet open" % (this_robot.name, this_robot.predicted_gain))
        return RobotBidResponse(False)

    output("Robot %s with predicted gain %f joined the window." % (this_robot.name, this_robot.predicted_gain))
    bidding_robots.append(this_robot)
    
    while not winner:
        if rospy.is_shutdown():
            sys.exit()

    if this_robot == winner:
        return RobotBidResponse(True)

    return RobotBidResponse(False)


def bid_for_human_server():
    rospy.Service('bid', RobotBid, bid_for_human)

def release_human_server():
    rospy.Service('release', HumanRelease, human_release)

if __name__ == "__main__":
    rospy.init_node('human_allocator')

    output('Window size: %f' % window_size)
    bid_for_human_server()
    release_human_server()

    while not rospy.is_shutdown():
        if not human_busy:
            winner = None
            bidding_robots = []
            output("-" * 100)
            output("New window created")
            accept_new_robots = True

            while not len(bidding_robots):
                if rospy.is_shutdown():
                    rospy.signal_shutdown("")

            current_window_start_time = datetime.now()
            current_window_end_time = current_window_start_time + timedelta(seconds=window_size)

            while datetime.now() < current_window_end_time:
                if rospy.is_shutdown():
                    rospy.signal_shutdown("")

            accept_new_robots = False
            output("Window closed.")
            time.sleep(0.1)

            if len(bidding_robots):
                winner = max(bidding_robots, key=lambda robot: robot.predicted_gain)
                output("Winner of the bidding is robot %s with predicted gain %f" % (winner.name, winner.predicted_gain))
                human_busy = True
            else:
                output("No robots participated in this window.")

            output("-" * 100)
            output("\n")
