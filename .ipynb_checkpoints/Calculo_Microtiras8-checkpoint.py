import numpy as np
import skrf as rf

# Definición de funciones auxiliares
def acot(x):
    return np.arctan(1 / x)

def w_microstrip(e_r, H, t, Z, f):
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
        e_rp = ((e_r + 1) / 2) + ((e_r - 1) / 2) * ((1 / np.sqrt(1 + (12 * H) / W)) + 0.004 * (1 - W_H) ** 2)
        Zo = (60 / np.sqrt(e_rp)) * np.log((8 * H / W) + (W / (4 * H)))

    Lambda_0 = 3e8 / f
    Lambda_p = Lambda_0 / np.sqrt(e_rp)

    return We, Zo, Lambda_p

# Parámetros
f = 1.8e9
Zo = 50
VCE = 1
IC = 60
e_r = 4.5 #permitividad relativa
H = 1.66e-3
t = 0.04e-3 #espesor del conductor

# Cargar archivo S2P
file = './BFP450/' + f'BFP450_w_noise_VCE_{VCE:.1f}V_IC_{IC:.0f}mA.s2p'
ntwk = rf.Network(file)
i = (np.abs(ntwk.f - f)).argmin()
S = ntwk.s[i]

# Estabilidad
Delta = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
abs_Delta = np.abs(Delta)
k = (1 - np.abs(S[0, 0]) ** 2 - np.abs(S[1, 1]) ** 2 + abs_Delta ** 2) / (
    2 * np.abs(S[0, 1] * S[1, 0])
)

# Cálculos de reflexión
B1 = 1 + np.abs(S[0, 0]) ** 2 - np.abs(S[1, 1]) ** 2 - abs_Delta ** 2
B2 = 1 + np.abs(S[1, 1]) ** 2 - np.abs(S[0, 0]) ** 2 - abs_Delta ** 2
C1 = S[0, 0] - Delta * np.conj(S[1, 1])
C2 = S[1, 1] - Delta * np.conj(S[0, 0])

# Coefciientes de reflexión adaptados
r_Ms = (B1 - np.sqrt(B1 ** 2 - 4 * np.abs(C1) ** 2)) / (2 * C1) if B1 > 0 else (B1 + np.sqrt(B1 ** 2 - 4 * np.abs(C1) ** 2)) / (2 * C1)
r_ML = (B2 - np.sqrt(B2 ** 2 - 4 * np.abs(C2) ** 2)) / (2 * C2) if B2 > 0 else (B2 + np.sqrt(B2 ** 2 - 4 * np.abs(C2) ** 2)) / (2 * C2)

# Impedancias complejas de entrada y salida
r_in = np.conj(r_Ms)
r_out = np.conj(r_ML)



# Ganancia máxima de transferencia
G_Tmax = 10*np.log10(np.abs(S[1,0])/np.abs(S[0,1])*k-np.sqrt(k**2-1))
print(G_Tmax)

# Impedancias
Z_in = Zo * (1 + r_in) / (1 - r_in)
Z_out = Zo * (1 + r_out) / (1 - r_out)
Z_s = np.conj(Z_in)
Z_L = np.conj(Z_out)

# Cálculo de microtiras
W_50, Z_50, Lambda_p_50 = w_microstrip(e_r, H, t, Zo, f)
QWT_50 = Lambda_p_50 / 4
# Acoplador de entrada
R_in = np.real(Z_in)
X_in = np.imag(Z_in)
R_inp = R_in * (1 + (X_in / R_in) ** 2)
X_inp = R_in * (R_inp / X_in)
Z1 = np.sqrt(R_inp * Zo)
w_in, imp_in, Lambda_p_in = w_microstrip(e_r, H, t, Z1, f)
Largo_ac_in = Lambda_p_in / 4
cap_in = 1 / (2 * np.pi * f * X_inp)
beta_in = (2 * np.pi) / Lambda_p_in
d_cap_in = (1 / beta_in) * acot(X_inp / Zo)

# Acoplador de salida
R_out = np.real(Z_out)
X_out = np.imag(Z_out)
Z2 = np.sqrt(R_out * Zo)
w_out, imp_out, Lambda_p_out = w_microstrip(e_r, H, t, Z2, f)
Largo_ac_out = Lambda_p_out / 4
X_out_2 = -(Zo ** 2) / X_out
cap_out = 1 / (2 * np.pi * f * X_out_2)
beta_out = (2 * np.pi) / Lambda_p_out
d_cap_out = (1 / beta_out) * acot(X_out_2 / Zo)

# Microstrip de 80 ohm
W_80, Z_80, Lambda_p_80 = w_microstrip(e_r, H, t, 80, f)
QWT_80 = Lambda_p_80 / 4

# Microstrip de 25 ohm
W_25, Z_25, Lambda_p_25 = w_microstrip(e_r, H, t, 25, f)
QWT_25 = Lambda_p_25 / 4

import pandas as pd
results = pd.DataFrame({
    "Parámetro": [
        "W_50", "Z_50", "Lambda_p_50",
        "Largo_ac_in", "cap_in", "d_cap_in",
        "Largo_ac_out", "cap_out", "d_cap_out",
        "W_80", "Z_80", "QWT_80",
        "W_25", "Z_25", "QWT_25", "QWT_50", "w_in", "w_out", "imp_in", "imp_out"
    ],
    "Valor": [
        W_50, Z_50, Lambda_p_50,
        Largo_ac_in, cap_in, d_cap_in,
        Largo_ac_out, cap_out, d_cap_out,
        W_80, Z_80, QWT_80,
        W_25, Z_25, QWT_25, QWT_50, w_in, w_out, imp_in, imp_out
    ]
})
print("\nResultados Microtiras:")
print(results)
print("hola")
