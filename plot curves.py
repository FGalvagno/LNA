import os
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import re
import plotly.graph_objects as go
folder = './BFP450/'
files = [f for f in os.listdir(folder) if f.endswith('.s2p')]

target_freq = 1.85e9  # Hz

data = []

# Collect data from all files
for fname in files:
    match = re.search(r'VCE_(\d+\.?\d*)V_IC_(\d+)mA', fname)
    if match:
        VCE = float(match.group(1))
        IC = int(match.group(2))

        ntwk = rf.Network(os.path.join(folder, fname))
        i = (np.abs(ntwk.f - target_freq)).argmin()
        S = ntwk.s[i]

        Delta = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        abs_Delta = np.abs(Delta)
        k = (1 - np.abs(S[0, 0])**2 - np.abs(S[1, 1])**2 + abs_Delta**2) / (
            2 * np.abs(S[0, 1] * S[1, 0]))

        data.append((VCE, IC, k))

# Convert to numpy arrays
data = np.array(data)
vce_vals = np.unique(data[:, 0])
ic_vals = np.unique(data[:, 1])
VCE_grid, IC_grid = np.meshgrid(vce_vals, ic_vals)

# Initialize Z grid
K_grid = np.full_like(VCE_grid, np.nan, dtype=float)

# Fill K values into the grid
for vce, ic, k in data:
    i = np.where(ic_vals == ic)[0][0]
    j = np.where(vce_vals == vce)[0][0]
    K_grid[i, j] = k

fig = go.Figure()

text = np.round(K_grid, 4).astype(str)

# Add heatmap (equivalent to pcolormesh)
fig.add_trace(go.Heatmap(
    x=VCE_grid[0],  # Assuming VCE_grid is a meshgrid
    y=IC_grid[:, 0],
    z=K_grid,
    text = text,
    texttemplate = "%{text}",
    colorscale='Viridis',
    zmin=0.8,
    zmax=1.1,
    colorbar=dict(title='Stability Factor k')
))

# Add contour for k = 1
fig.add_trace(go.Contour(
    x=VCE_grid[0],
    y=IC_grid[:, 0],
    z=K_grid,
    contours=dict(
        start=1,
        end=1,
        size=1,
        coloring='none'
    ),
    line=dict(color='red', width=2),
    showscale=False
))
x_vals = np.linspace(0, 4)
y_vals = 280 * x_vals        #CAMBIAR ESTO POR EL LIMITE DEL TRANSISTOR QUE QUEREMOS USAR

fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='lines',
    line=dict(color='green', width=2, dash='dot'),
    name='y = 280'
))



# Update layout
fig.update_layout(
    title=f'Pseudocolor Plot of Stability Factor k at {target_freq / 1e9:.1f} GHz',
    xaxis=dict(title='VCE (V)', tickvals=vce_vals, gridcolor='rgba(0,0,0,0.3)', showgrid=True, range=[0.75, 4.25]),
    yaxis=dict(title='IC (mA)', tickvals=ic_vals, gridcolor='rgba(0,0,0,0.3)', showgrid=True, range=[9, 91]),
    plot_bgcolor='white',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

fig.show()
fig.write_html('BFP_450.html')

