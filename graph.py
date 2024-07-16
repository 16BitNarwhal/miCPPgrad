import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

data = pd.read_csv('out/neural_network_3d_graph.csv')

epochs = data['Epoch'].unique()

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

X = data[data['Epoch'] == epochs[0]]['X'].unique()
Y = data[data['Epoch'] == epochs[0]]['Y'].unique()
X, Y = np.meshgrid(X, Y)
Z_pred = np.zeros_like(X)
Z_true = data[data['Epoch'] == epochs[0]]['Z_true'].values.reshape(X.shape)

surf_pred = ax1.plot_surface(X, Y, Z_pred, cmap='viridis')
surf_true = ax2.plot_surface(X, Y, Z_true, cmap='viridis')

for ax in [ax1, ax2]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(0, 2)

title_pred = ax1.set_title('')
title_true = ax2.set_title('Ground Truth')

def update_plot(frame):
    epoch = epochs[frame]
    epoch_data = data[data['Epoch'] == epoch]

    Z_pred = epoch_data['Z_pred'].values.reshape(X.shape)
    Z_true = epoch_data['Z_true'].values.reshape(X.shape)

    ax1.clear()
    surf_pred = ax1.plot_surface(X, Y, Z_pred, cmap='viridis')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_zlim(0, 2)
    ax1.set_title(f'Prediction (Epoch {epoch}, Loss: {epoch_data["Loss"].iloc[0]:.4f})')

    ax2.clear()
    surf_true = ax2.plot_surface(X, Y, Z_true, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_zlim(0, 2)
    ax2.set_title('Ground Truth')
    
    return surf_pred, surf_true

anim = animation.FuncAnimation(fig, update_plot, frames=len(epochs), interval=200, blit=False)
anim.save('out/neural_network_3d_graph.gif', writer='pillow', fps=5)

plt.close(fig)
print("Animation saved as 'out/neural_network_3d_graph.gif'")