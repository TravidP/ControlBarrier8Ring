import numpy as np
import matplotlib.pyplot as plt

R = 250.0
t_vals = np.linspace(0, 2 * np.pi, 10000)

x_base = -R * np.sin(t_vals - 6)
y_base = -R * np.sin(2 * t_vals)

# One inward-then-outward wave per loop, centred near (≈222, 100) / (≈-222, 100)
A = 90.0    # wave amplitude in metres
w = 0.14    # width of each Gaussian lobe
delta = 0.26  # half-distance between inward and outward lobes
t_r = 1.77              # right loop centre (x≈+222, y≈+100)
t_l = t_r + np.pi       # left  loop centre, symmetric

def _g(u):
    return np.exp(-u**2 / (2 * w**2))

# Right loop: inward (-x) before t_r, outward (+x) after t_r
# Left  loop: inward (+x) before t_l, outward (-x) after t_l  (opposite sign)
dent = A * (
    _g(t_vals - t_r - delta/2) - _g(t_vals - t_r + delta/2)   # right
  + _g(t_vals - t_l + delta/2) - _g(t_vals - t_l - delta/2)   # left
)

x_vals = x_base + dent
y_vals = y_base

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_vals, y_vals, linewidth=2)
ax.set_aspect('equal')
ax.set_title(f'Figure-8 track  (R = {R} m)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid(True)
plt.tight_layout()
plt.show()
