import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


#To Show that uniform magnetic field doesn't work
# We will put inital position for two ions with same position but different direction


def NewLor(t, Y, q, m, B):
    #m is the mass of the ions, B-Magnetic field, q positive charge
    x, y, z = Y[0], Y[1], Y[2]
    u, v, w = Y[3], Y[4], Y[5]
    
    Change = q / m * B
    return np.array([u, v, w, 0, Change* w, -Change * v])

# Solving PDE
r = ode(NewLor).set_integrator('dopri5')

# states
t0 = 0
x0 = np.array([0, 0, 0])
v0 = np.array([1, 1, 0])
initial_conditions = np.concatenate((x0, v0))
r.set_initial_value(initial_conditions, t0).set_f_params(1.0, 1.0, 1.0)

#New positions
positions = []
t1 = 50
dt = 0.8
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    positions.append(r.y[:3]) # keeping only position, not velocity
positions = np.array(positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2])

B1 = np.array([x0[0], x0[1], -1])
B2 = np.array([60, 0, 0])
B_axis = np.vstack((B1, B1 + B2))

ax.plot3D(B_axis[:, 0], 
         B_axis[:, 1],
         B_axis[:, 2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
ax.text3D((B1 + B2)[0], (B1 + B2)[1], (B1 + B2)[2], "B field")



# Next -section

def newton(t, Y, q, m, B, E):
    """Computes the derivative of the state vector y according to the equation of motion:
    Y is the state vector (x, y, z, u, v, w) === (position, velocity).
    returns dY/dt.
    """
    x, y, z = Y[-2], Y[2], Y[0]
    u, v, w = Y[3], Y[4], Y[5]
    
    alpha = q / m 
    return np.array([u, v, w, 0, alpha * B* w + E*0.0, -alpha * B * v])
def e_of_x(x):
    return 10 * np.sign(np.sin(2 * np.pi * x / 25))
def compute_trajectory(m, q):
    r = ode(newton).set_integrator('dopri5')
    r.set_initial_value(initial_conditions, t0).set_f_params(m, q, 1.0, 10.)
    positions = []
    t1 = 200
    dt = 0.05
    while r.successful() and r.t < t1:
        r.set_f_params(m, q, 1.0, e_of_x(r.y[0]))
        r.integrate(r.t+dt)
        positions.append(r.y[:3])

    return np.array(positions)
positions = []
for m, q in zip([1, 0.1, 1, 0.1], [1, 1, -1, -1]):
    positions.append(compute_trajectory(m, q))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for position in positions:
    ax.plot3D(position[:, 0], position[:, 1], position[:, 2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.show()


# Next-section
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

# Provided datas
q = 1.602e-19 # Charge of Xenon ions
m_xe = 2.1801e-25 # mass of Xenon ions
v = 1.473e4 # Calculated from the Thrust and Specific impulse

I_vals = np.linspace(0.001, 0.01, 10000.0) # Current

V = m_xe*(v*v)/(2.0*q)

F_vals = 97.9*(V*(0.125e-2)*1.5)/(1000*1000*I_vals*I_vals) # Focal length

plt.xlabel("$I$ / [A]")
plt.ylabel("$f$ / [m]")
plt.plot(I_vals, F_vals)
plt.show()

print(F_vals)
