import numpy as np
from math import sin,cos,atan2 as arctan

from .parameter_configure import get_Uplink_param

class Attacker_Uplink():
    def __init__(self,uav_weight,delta_t,param_id=0):
        self.delta_t = delta_t
        self.uav_weight = uav_weight
        self.phi, self.V_a, self.W_a, self.zeta = get_Uplink_param(param_id)
        self.re_phi = np.linalg.inv(self.phi)
        self.K_p = np.diag([-12, -12, -35])
        self.K_v = np.diag([-8, -8, -18])
        self.c_iter = 0
        self.g = np.array([0,0,9.8])

    def get_re_attacked_position(self,position):
        chi_a = np.dot(self.V_a, self.zeta).reshape(-1)
        pos_a = np.dot(self.re_phi, position - chi_a)
        return pos_a

    def get_attacked_position(self,position):
        chi_a = np.dot(self.V_a, self.zeta)
        pos_a = np.dot(self.phi, position) + chi_a
        return pos_a.reshape(-1)

    def calc_F_a(self,
                 e_p, e_v,
                 p_d, p_d_dot, p_d_dot2
                 ):
        F = np.dot(self.uav_weight, np.dot(self.K_p, e_p) + \
                   np.dot(self.K_v, e_v) + p_d_dot2) + self.g * self.uav_weight
        F_phi = np.dot((self.phi - np.eye(3)),
                       (-np.dot(self.K_p, p_d) - np.dot(self.K_v, p_d_dot) + p_d_dot2))
        F_chi_a = np.dot(-np.dot(self.K_p, self.V_a) -
                         np.dot(np.dot(self.K_v, self.V_a), self.W_a) +
                         np.dot(np.dot(self.V_a, self.W_a), self.W_a), self.zeta)
        F_a = F + self.uav_weight * F_phi + self.uav_weight * F_chi_a
        return F_a

    def calc_attitude(self,F):
        psi = 0
        theta_a = arctan(F[0] * cos(psi) + F[1] * sin(psi), F[2])
        phi_a = arctan(F[0] * sin(psi) - F[1] * cos(psi),
                       F[2] / cos(theta_a))
        attitude = np.array([phi_a, theta_a, psi])
        return attitude

    def execute(self,cur_real_state,cur_desired_state,nxt_desired_state):
        cur_pos = cur_real_state[1:4]
        cur_vel = cur_real_state[4:7]

        cur_d_pos = cur_desired_state[1:4]
        cur_d_vel = cur_desired_state[4:7]

        nxt_d_pos = nxt_desired_state[1:4]
        # nxt_d_pos_dot = nxt_desired_state[4:7]
        # nxt_d_pos_dot2 = nxt_desired_state[7:10]
        nxt_d_pos_dot2 = 2 * ((nxt_d_pos - cur_pos) - cur_vel * self.delta_t) / self.delta_t ** 2
        nxt_d_pos_dot = cur_vel + nxt_d_pos_dot2 * self.delta_t

        e_p = cur_pos - cur_d_pos
        e_v = cur_vel - cur_d_vel

        F_a = self.calc_F_a(e_p, e_v, nxt_d_pos, nxt_d_pos_dot, nxt_d_pos_dot2)
        nxt_attitude = self.calc_attitude(F_a)
        nxt_acc = F_a / self.uav_weight - self.g
        nxt_vel = cur_vel + nxt_acc * self.delta_t
        s = (nxt_vel**2 - cur_vel**2) / (2 * nxt_acc)
        nxt_pos = cur_pos + s
        nxt_pos[2] = 1.
        return np.concatenate([nxt_pos,nxt_vel,nxt_acc,nxt_attitude],axis=0)

    def execute_after_compensation(self,cur_real_state,nxt_position,fly_by_command_func):
        cur_position = cur_real_state[[1,2,3]]
        cur_velocity = cur_real_state[[4,5,6]]
        nxt_attacked_position = self.get_attacked_position(nxt_position)

        nxt_real_state = fly_by_command_func(cur_position,cur_velocity,nxt_attacked_position)
        return nxt_real_state

    def update(self,):
        self.zeta_a_dot = np.dot(self.W_a, self.zeta)
        self.zeta += self.zeta_a_dot * self.delta_t
        self.c_iter += 1