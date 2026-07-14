import numpy as np
import matplotlib.pyplot as plt

R = 250.0
t_vals = np.linspace(0, 2 * np.pi, 10000)

x_vals = -R * np.sin(t_vals - 6)
y_vals = -R * np.sin(2 * t_vals) 

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_vals, y_vals, linewidth=2)
ax.set_aspect('equal')
ax.set_title(f'Figure-8 track  (R = {R} m)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid(True)
plt.tight_layout()
plt.show()
