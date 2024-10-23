from em_vehicle_control.em_vehicle_control.helper_classes.mpc_tracker_theta import *
from helper_classes.segment import *
from em_vehicle_control_msgs.msg import Pose2D, Path2D


mpc_tracker = MPCTracker()
path = [

    ]
current_pose = (-3.8000, 2.0000, 0)
x_ref, y_ref, theta_ref, direction_ref = mpc_tracker.get_reference_path(
        current_pose, path
    )

