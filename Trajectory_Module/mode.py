class Trajectory():
    def __init__(self,configure):
        self.period = configure["period"]
        self.radius = configure["raidus"]
        self.delta_t = configure["delta_t"]
        self.start_time = configure["start_time"]

class UAV():
    def __init__(self,configure):
        self.uav_num = configure["uav_num"]
        self.offsets = configure["offsets"]
        self.uav_weight = configure["weight"]
        self.limit = configure["limit"]