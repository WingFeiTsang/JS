import numpy as np

def get_Uplink_param(param_id=0):
    phi_att = np.diag([0.5, 0.6, 1])
    V_a = np.array([[0.5, 0], [0.6, 0], [0, 0]])
    W_a = np.array([[0, 1], [-1, 0]])
    zeta = np.array([0.1, 0.1])
    return phi_att, V_a, W_a, zeta

def get_Downlink_param(param_id=0):
    ya0 = np.array([
         -0.18,  -0.17, 0,
         -0.14,  -0.13, 0,
         -0.13,  -0.12, 0,
        -0.013, -0.012, 0,
    ])
    ya1 = np.array([
        0.18,  0.17,  0,
        0.14,  0.13,  0,
        0.13,  0.12,  0,
        0.013, 0.012, 0,
    ])
    y_change = np.array([
         -0.002,   0.003, 0,
         -0.004, -0.0032, 0,
        -0.0012, -0.0009, 0,
         -0.004,  -0.001, 0,
    ])
    return ya0, ya1, y_change