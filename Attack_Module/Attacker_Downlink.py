import copy

from .parameter_configure import get_Downlink_param

class Attacker_Downlink():
    def __init__(self,param_id=0):
        y0,y1,self.y_change = get_Downlink_param(param_id)
        self.y = [y0,y1]
        self.iter = 0

    def execute(self,single_state):
        after_attack = copy.deepcopy(single_state[1:13])
        after_attack += self.y[self.iter % len(self.y)]
        return after_attack

    def update(self):
        self.y[self.iter % len(self.y)] += self.y_change
        self.iter += 1