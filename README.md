# AMR Final Project — 3-Wheel Differential Drive Robot

## Overview
This project focuses on the modeling, control, and motion planning of a **3-wheeled differential drive mobile robot**, consisting of **two actively driven wheels and one passive caster wheel**.  
The work demonstrates how kinematic and dynamic models are integrated with control strategies and motion planning techniques to achieve stable navigation and obstacle avoidance.

---

## 1. System Modeling

### Kinematics
- Developed the kinematic model of a differential drive robot
- Derived forward and inverse kinematics relating wheel velocities to robot motion
- Modeled the robot as a non-holonomic system operating in a planar environment

### Dynamics
- Implemented simplified wheel and motor dynamics
- Incorporated actuator dynamics for realistic velocity tracking
- Combined dynamics with kinematics for simulation and control

### Physical Working Principles and Implementation
- Two independently driven wheels generate linear and angular motion
- A passive caster wheel provides balance without affecting kinematics
- Models implemented in Python and validated through simulation

---

## 2. Control Methodology

### Control Approach
- Implemented PID controllers for wheel velocity tracking
- Converted desired linear and angular velocities into wheel commands using inverse kinematics
- Ensured smooth and stable motion for both straight-line and circular trajectories

### Stability Analysis
- Analyzed closed-loop behavior using velocity error dynamics
- Demonstrated asymptotic stability of the PID-controlled system
- Ensured bounded errors under realistic actuator constraints

### Testing and Validation
- Tested straight-line motion (constant linear velocity, zero angular velocity)
- Tested circular motion (constant linear and angular velocity)
- Validated controller performance through trajectory tracking and animations

---

## 3. Motion Planning

### Planning Algorithms and Approach
- Implemented artificial potential field–based motion planning
- Combined attractive forces toward the goal and repulsive forces from obstacles
- Added tangential (swirl) components to avoid local minima

### Obstacle Avoidance
- Modeled obstacles with safety (inflated) no-go zones
- Ensured collision-free navigation around multiple obstacles
- Integrated planning with differential drive constraints

### Key Principles
- Real-time reactive planning
- Smooth and continuous trajectories
- Compatibility with non-holonomic motion constraints

---

## Results
- Successfully navigated the robot from start to goal positions
- Achieved stable straight-line and circular motion tracking
- Demonstrated reliable obstacle avoidance with two obstacles
- Animations confirm smooth trajectories, stable control, and goal convergence

---

## Challenges and Solutions
- **Local minima in potential fields:** mitigated using tangential (swirl) forces
- **Oscillations near the goal:** addressed by introducing a final-approach mode
- **Over-aggressive obstacle repulsion:** resolved by capping repulsive forces and inflating obstacles consistently

---
## Files Description

- **AMR_Final_DevamShah.pptx**  
  Final presentation covering system modeling, control methodology, motion planning, results, and challenges.

- **kinematicslinear.py**  
  Straight-line motion simulation of a 3-wheeled differential drive robot  
  *(linear velocity v = x, angular velocity ω = 0)*.  
  Includes kinematics, wheel dynamics, PID control, and animation.

- **kinematicsangular.py**  
  Circular motion simulation of a 3-wheeled differential drive robot  
  *(linear velocity v = x, angular velocity ω = y)*.  
  Includes kinematics, dynamics, PID control, and animation.

- **ObstacleAvoidance.py**  
  Motion planning and obstacle avoidance simulation using artificial potential fields.  
  Demonstrates navigation around two obstacles with safety (no-go) zones and animation.

- **tests_and_simulations/**  
  Contains simulation videos demonstrating straight-line motion, circular motion, and obstacle avoidance.  
  *(Videos are hosted externally due to GitHub file size limitations.)*

- **requirements.txt**  
  Lists the Python dependencies required to run the simulations.


## Setup
```bash
pip install -r requirements.txt
