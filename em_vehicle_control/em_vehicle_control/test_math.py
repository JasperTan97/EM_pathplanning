from helper_classes.map import RoadMap, RoadSegment, RoadGraph, RoadTrack
from helper_classes.path_planner import PathPlanner
from helper_classes.vehicles import EdyMobile, Edison
from helper_classes.pathplanners.rrt import *
from helper_classes.pathplanners.pp_viz import RoadMapAnimator
from shapely import within
import time
import networkx as nx
import numpy as np

#############################
# Check calc_theta: ??????? #
#############################
def calc_theta():
    x = np.array([0, 1, 0, -1, 0])
    y = np.array([0, 1, 2, 1, 0])
    dx = np.diff(x)
    dy = np.diff(y)
    angles = np.arctan2(dy, dx)
    print(angles)
    print(np.degrees(angles))


##########################
# Check tracker: ??????? #
##########################
def check_tracker():
    direction = 1
    oe = 0
    cte = 0.1

    orientation_error_bounds = 10 / 180 * np.pi  # radians
    K_omega = 2.0  # propotional gain for turning speed
    v_max = 0.1  # m/s, maximum allowable forward speed
    omega_max = 0.2  # rad/s
    k_v = 0.5  # gain for reducing linear speed when far from track
    k_w = 5.0  # steepness of the blending transition
    d_blend = 0.5  # blending threshold for cross-track error


    if direction == 1:
        desired_velocity = v_max
        velocity_sign = 1
    else:
        desired_velocity = -v_max
        velocity_sign = -1

    # Orientation adjustment
    if not -orientation_error_bounds < oe < orientation_error_bounds:
        omega = K_omega * oe
        omega = np.clip(omega, -omega_max, omega_max)
        v = 0.0
    else:
        w = 1 / (1 / np.exp(-k_w * cte - d_blend))
        v_move = v_max * np.exp(-k_v * cte)
        v_move *= velocity_sign  # Adjust based on direction
        omega = K_omega * oe
        omega = np.clip(omega, -omega_max, omega_max)
        v = w * v_move + (1 - w) * desired_velocity

        print(v)

################################
# Check OE calculator: WORKING #
################################
def check_oe():
    lap = np.array([-3.7977, 1.9966])
    car = np.array([-3.799, 2.000])
    car_yaw = 0.1
    neg_lap = lap - 2*(lap-car)
    neg_lap = np.array([-3.7785, 2.002])

    tla = np.arctan2(lap[1]-car[1], lap[0]-car[0])
    neg_tla = np.arctan2(neg_lap[1]-car[1], neg_lap[0]-car[0])

    # print(tla, neg_tla)

    calc_oe = lambda tla:  (tla - car_yaw + np.pi) % (2*np.pi) - np.pi
    print(calc_oe(tla), calc_oe(neg_tla + np.pi))

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot the Car Position
    plt.plot(car[0], car[1], 'bo', markersize=10, label='Car Position')

    # Plot the Negative Lap Point
    plt.plot(neg_lap[0], neg_lap[1], 'ro', markersize=8, label='Negative Lap Point')

    # Draw Car Yaw as an Arrow
    # Define arrow properties
    arrow_length = 0.05  # Length of the arrow
    dx = arrow_length * np.cos(car_yaw)  # X component
    dy = arrow_length * np.sin(car_yaw)  # Y component

    plt.arrow(
        car[0], car[1], dx, dy,
        head_width=0.01, head_length=0.01,
        fc='blue', ec='blue',
        length_includes_head=True,
        label='Car Yaw'
    )

    # Optionally, draw the negative yaw for better visualization
    # Uncomment the following lines if desired
    # neg_car_yaw = car_yaw + np.pi
    # neg_car_yaw = (neg_car_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize
    # dx_neg = arrow_length * np.cos(neg_car_yaw)
    # dy_neg = arrow_length * np.sin(neg_car_yaw)
    # plt.arrow(
    #     car[0], car[1], dx_neg, dy_neg,
    #     head_width=0.1, head_length=0.1,
    #     fc='cyan', ec='cyan',
    #     length_includes_head=True,
    #     label='Negative Car Yaw'
    # )

    # Adding Labels and Title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Position and Orientation Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling for both axes

    # Display the Plot
    plt.show()

if __name__=="__main__":
    # check_oe()
    # check_tracker()
    calc_theta()