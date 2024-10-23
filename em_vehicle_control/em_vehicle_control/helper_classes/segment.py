from shapely import LineString
from typing import List
from em_vehicle_control_msgs.msg import Pose2D

class Segment:
    def __init__(self, start: Pose2D, end: Pose2D):
        self.start = start
        self.end = end
        self.line = LineString([(start.x, start.y), (end.x, end.y)])
        if start.direction_flag == Pose2D.BACKWARD:
            self.direction = -1
        else:
            self.direction = 1
        self.length = 0
        self.length = self.line.length

    def __repr__(self) -> str:
        return f"{self.line}, {self.direction} with length {self.length}"
    
    def re_init(self):
        self.line = LineString([(self.start.x, self.start.y), (self.end.x, self.end.y)])
        self.length = self.line.length
    
def create_segments(path: List[Pose2D]) -> List[Segment]:
        segments = []
        for i in range(len(path) - 1):
            segments.append(Segment(path[i], path[i + 1]))
        return segments