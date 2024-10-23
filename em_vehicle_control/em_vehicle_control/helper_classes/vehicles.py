from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from typing import Tuple, List, Optional

class BaseVehicle():
    def __init__(
            self, 
            start_position: List,
            min_wheel_acceleration: float,
            max_wheel_acceleration: float,
            min_wheel_speed: float,
            max_wheel_speed: float,
            wheel_radius: float,
            length_of_wheel_axis: float
        ):
        self._x = start_position[0]
        self._y = start_position[1]
        self._theta = start_position[2] # theta is measured clockwise from the x axis
        self._min_wheel_acceleration = min_wheel_acceleration
        self._max_wheel_acceleration = max_wheel_acceleration
        self._min_wheel_speed = min_wheel_speed
        self._max_wheel_speed = max_wheel_speed
        self._wheel_radius = wheel_radius
        self._length_of_wheel_axis = length_of_wheel_axis
        self._L_wheel_speed = 0
        self._R_wheel_speed = 0
        self._vehicle_model = None
        self._linear_velocity = None
        self._angular_velocity = None

    def visualise(self) -> None:
        """Get shape of vehicle"""
        x, y = self._vehicle_model.exterior.xy
        mpl_polygon = MplPolygon(list(zip(x, y)), closed=True, edgecolor='black', facecolor='lightblue', alpha=0.5)
        _, ax = plt.subplots()
        ax.add_patch(mpl_polygon)
        ax.axis('scaled')
        plt.show()

    def step(self, dt: float, left_acc: float, right_acc: float) -> None:
        """Updates the state of the vehicle after timestep dt with input left_acc and right_acc"""
        left_acc = np.clip(left_acc, self._min_wheel_acceleration, self._max_wheel_acceleration)
        right_acc = np.clip(right_acc, self._min_wheel_acceleration, self._max_wheel_acceleration)
        # step in wheel velocity
        self._L_wheel_speed = self._L_wheel_speed + left_acc * dt
        self._L_wheel_speed = np.clip(self._L_wheel_speed, self._min_wheel_speed, self._max_wheel_speed)
        self._R_wheel_speed = self._R_wheel_speed + right_acc * dt
        self._R_wheel_speed = np.clip(self._R_wheel_speed, self._min_wheel_speed, self._max_wheel_speed)
        # wheel velocities are assumed to be held constant throughout dt
        self._linear_velocity = self._wheel_radius * (self._L_wheel_speed + self._R_wheel_speed) / 2
        self._angular_velocity = self._wheel_radius * (self._R_wheel_speed - self._L_wheel_speed) / self._length_of_wheel_axis
        # compute new position of the vehicle, in dt, we move linearly followed by rotation
        self._x = self._x + self._linear_velocity * dt * np.cos(self._theta)
        self._y = self._y + self._linear_velocity * dt * np.sin(self._theta)
        self._theta = (self._theta + self._angular_velocity * dt) % (2*np.pi)
    
    def get_state(self) -> Tuple:
        return self._x, self._y, self._theta, self._linear_velocity, self._angular_velocity

class EdyMobile(BaseVehicle):
    """
    To use, initialise the start posiiton
    """
    """EdyMobile datasheet"""
    def __init__(self, start_position: Optional[Tuple[float, float, float]] = [0,0,0]):
        """
        start_position: x, y, theta values of the vehicle
        """
        # Define the 'centre' of the EdyMobile as centre of the wheel axis
        self._length_of_wheel_axis = 0.18
        self._length_to_back_rect = 0.03
        self._length_to_front_rect = 0.11
        # Define wheel properties
        self._min_wheel_acceleration = -0.2 # rad/s/s
        self._max_wheel_acceleration = 0.2
        self._min_wheel_speed = -0.5 # rad/s
        self._max_wheel_speed = 0.5
        self._wheel_radius = 0.08
        self.turning_radius = self._length_of_wheel_axis/2

        super().__init__(
            start_position, 
            self._min_wheel_acceleration, 
            self._max_wheel_acceleration,
            self._min_wheel_speed,
            self._max_wheel_speed,
            self._wheel_radius,
            self._length_of_wheel_axis
        )

        # construct EdyMobile shape
        self.construct_vehicle()

    def construct_vehicle(self, state: Optional[Tuple[float, float, float]] = None):
        """Reconstructs the shapely object representing the vehicle"""
        if state is not None:
            self._x, self._y, self._theta = state
        radius = self._length_of_wheel_axis / 2
        frontct = self._length_to_front_rect * np.cos(self._theta)
        frontst = self._length_to_front_rect * np.sin(self._theta)
        backct = self._length_to_back_rect * np.cos(self._theta)
        backst = self._length_to_back_rect * np.sin(self._theta)
        radct = radius * np.cos(self._theta)
        radst = radius * np.sin(self._theta)
        center = Point(self._x + frontct, self._y + frontst)
        circle = center.buffer(radius, resolution=100)
        rectangle = Polygon([
            (self._x + frontct + radst, self._y + frontst - radct),
            (self._x + frontct - radst, self._y + frontst + radct),
            (self._x - backct - radst, self._y - backst + radct),
            (self._x - backct + radst, self._y - backst - radct)
        ])
        self._vehicle_model = unary_union([rectangle, circle])

    def step(self, dt: float, left_acc: float, right_acc: float) -> None:
        super().step(dt, left_acc, right_acc)
        self.construct_vehicle()

    
class Edison(BaseVehicle):
    def __init__(self, start_position: Optional[Tuple[float, float, float]] = [0,0,0]):
        self._length_of_wheel_axis = 0.23
        self.body_radius = 0.245/2
        # Define wheel properties
        self._min_wheel_acceleration = -0.2 # rad/s/s
        self._max_wheel_acceleration = 0.2
        self._min_wheel_speed = -0.5 # rad/s
        self._max_wheel_speed = 0.5
        self._wheel_radius = 0.08
        self.turning_radius = self._length_of_wheel_axis/2

        super().__init__(
            start_position, 
            self._min_wheel_acceleration, 
            self._max_wheel_acceleration,
            self._min_wheel_speed,
            self._max_wheel_speed,
            self._wheel_radius,
            self._length_of_wheel_axis
        )
        self.construct_vehicle()

    def construct_vehicle(self, state: Optional[Tuple[float, float, float]] = None):
        """Reconstructs the shapely object representing the vehicle"""
        if state is not None:
            self._x, self._y, self._theta = state
        center = Point(self._x, self._y)
        circle = center.buffer(self.body_radius, resolution=100)
        self._vehicle_model = circle

    def step(self, dt: float, left_acc: float, right_acc: float) -> None:
        super().step(dt, left_acc, right_acc)
        self.construct_vehicle()