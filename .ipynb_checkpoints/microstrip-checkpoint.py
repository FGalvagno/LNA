import numpy as np

def w_microstrip(e_r, H, t, Z, f, QW = False):
    A = (Z / 60) * np.sqrt((e_r + 1) / 2) + ((e_r - 1) / (e_r + 1)) * (0.226 + (0.121 / e_r))
    B = (377 * np.pi) / (2 * Z * np.sqrt(e_r))

    W_H = (8 * np.exp(A)) / (np.exp(2 * A) - 2)

    if W_H > 2:
        W_H = (e_r-1)/ (np.pi*e_r) * (np.log(B - 1) + 0.293 - 0.517/e_r) + 2/np.pi *(B-1-np.log(2*B-1))

    W = W_H * H

    if W_H <= (1 / (2 * np.pi)):
        We = W + (t / np.pi) * (1 + np.log((4 * np.pi * W) / t))
    else:
        We = W + (t / np.pi) * (1 + np.log((2 * H) / t))

    if W_H >= 1:
        e_rp = ((e_r + 1) / 2) + ((e_r - 1) / 2) * (1 / np.sqrt(1 + (12 * H) / W))
        Zo = ((120 * np.pi) / np.sqrt(e_rp)) / (W_H + 2.46 - 0.49*(1/W_H) + (1-1/W_H)**6)
    else:
        e_rp = ((e_r + 1) / 2) + ((e_r - 1) / 2) * (
            (1 / np.sqrt(1 + (12 * H) / W)) + 0.004 * (1 - W_H) ** 2
        )
        Zo = (60 / np.sqrt(e_rp)) * np.log((8 * H / W) + (W / (4 * H)))

    Lambda_0 = 3e8 / f
    Lambda_p = Lambda_0 / np.sqrt(e_rp)

    if QW:
        return dict(We = We, Zo = Zo, Lambda_p = Lambda_p, d = Lambda_p/4)
    else:
        return dict(We = We, Zo = Zo, Lambda_p = Lambda_p)

def QW_microstrip(e_r, H, t, Z, f):
    return w_microstrip(e_r, H, t, Z, f, QW = True)
    