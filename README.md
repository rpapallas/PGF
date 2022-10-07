# The Predictive Guided Framework (PGF)

This is the source code for the Predictive Guided Framework. Please refer to the academic paper published at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2022: [To ask for help or not to ask: A predictive approach to human-in-the-loop motion planning for robot manipulation tasks](https://rpapallas.com/publications/iros-2022-pgf/).

## Abstract

We present a predictive system for non-prehensile, physics-based motion planning in clutter with a human-in- the-loop. Recent shared-autonomous systems present motion planning performance improvements when high-level reasoning is provided by a human. Humans are usually good at quickly identifying high-level actions in high-dimensional spaces, and robots are good at converting high-level actions into valid robot trajectories. In this paper, we present a novel framework that permits a single human operator to effectively guide a fleet of robots in a virtual warehouse. The robots are tackling the problem of Reaching Through Clutter (RTC), where they are reaching onto cluttered shelves to grasp a goal object while pushing other obstacles out of the way. We exploit information from the motion planning algorithm to predict which robot requires human help the most and assign that robot to the human. With twenty virtual robots and a single human-operator, the results suggest that this approach improves the system’s overall performance compared to a baseline with no predictions. The results also show that there is a cap on how many robots can effectively be guided simultaneously by a single human operator.

## Citation

If you used part of this code, please consider citing our work:

```
@inproceedings{papallas2022ask,
  title={To ask for help or not to ask: A predictive approach to human-in-the-loop motion planning for robot manipulation tasks},
  author={Papallas, Rafael and Dogar, Mehmet R},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2022)},
  year={2022},
  organization={IEEE}
}
```

```
Papallas, R. and Dogar, M.R., 2022, June. To ask for help or not to ask: A predictive approach to human-in-the-loop motion planning for robot manipulation tasks. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2022). IEEE.
```
