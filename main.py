import numpy as np
import plotly.graph_objects as go

# Physical Constants
MU_0 = 1  
R_1 = 2
R_2 = 12
I_1 = 15
I_2 = 11
N = 100  
N_field = 10  

# Compute segment properties
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
dL_1 = 2 * R_1 * np.sin(np.pi / N)
dL_2 = 2 * R_2 * np.sin(np.pi / N)

def ring_current(field_positions, R_ring, dL_ring, I, R):
    """
    Computes the magnetic field at each position due to the current loop.

    Parameters:
        field_positions (numpy array): The positions where the magnetic field is to be calculated (N_beam, 3).

    Returns:
        numpy array: The magnetic field at each position (N_beam, 3).
    """
    B_field = np.zeros_like(field_positions, dtype=np.float64)
    
    for i in range(N):
        r_field = field_positions - R_ring[i]  # Vector pointing to the field position from the current ring element
        r_field_mag = np.linalg.norm(r_field, axis=1)[:, np.newaxis]  # Magnitude of the r_field for each electron
        r_unit = r_field / r_field_mag  # Unit vector in the direction of r_field
        dB_contribution = np.cross(dL_ring[i], r_unit) / r_field_mag**2  # Magnetic field contribution from current element
        B_field += dB_contribution  # Sum contributions

    return (MU_0 * I) / (4 * np.pi) * B_field  # Apply the Biot-Savart constant

# Generate random vectors for electron positions within a cone
def generate_cone_vectors(N, alpha):
    """
    Generates random vectors within a cone defined by angle alpha.

    Parameters:
        N (int): Number of vectors to generate.
        alpha (float): Half-angle of the cone in radians.

    Returns:
        numpy array: The generated vectors (N, 3).
    """
    phi = np.random.uniform(0, 2 * np.pi, N)  # Azimuthal angle
    theta = np.random.uniform(0, alpha, N)  # Polar angle within the cone
    
    # Convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack((x, y, -z))

# Simulation parameters
N_beam = 100
beam_width = np.pi / 250
q = 0.25
m = 1
dt = 0.1
t = 0
tf = 350

# Set up 2 lenses
# Discretized positions of the current elements
R_ring_1 = np.column_stack((R_1 * np.cos(theta), R_1 * np.sin(theta), np.zeros(N)))
dL_ring_1 = dL_1 * np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(N)))

# Discretized positions of the current elements
R_ring_2 = np.column_stack((R_2 * np.cos(theta), R_2 * np.sin(theta), -30*np.ones(N)))
dL_ring_2 = dL_2 * np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(N)))


# Initialize positions and velocities of the electrons
x = np.array([np.array([0, 0, 10], dtype=np.float64) for _ in range(N_beam)], dtype=np.float64)
v = generate_cone_vectors(N=N_beam, alpha=beam_width)

# Track electron positions over time
x_t = []
while t < tf:
    B_total = ring_current(x, R_ring_1, dL_ring_1,I_1, R_1) + ring_current(x,R_ring_2, dL_ring_2, I_2, R_2)
    a = q / m * np.cross(v, B_total)  # Acceleration due to the magnetic field
    v += a * dt  # Update velocity
    x += v * dt  # Update position
    x_t.append(x.copy())
    t += dt


# Initialize positions and velocities of the electrons
x2 = np.array([np.array([0.1, 0, 10], dtype=np.float64) for _ in range(N_beam)], dtype=np.float64)
v2 = generate_cone_vectors(N=N_beam, alpha=beam_width)

# Track electron positions over time
x2_t = []
t=0
while t < tf:
    B_total = ring_current(x2, R_ring_1, dL_ring_1,I_1, R_1) + ring_current(x2,R_ring_2, dL_ring_2, I_2, R_2)
    a = q / m * np.cross(v2, B_total)  # Acceleration due to the magnetic field
    v2 += a * dt  # Update velocity
    x2 += v2 * dt  # Update position
    x2_t.append(x2.copy())
    t += dt

'''
# Prepare data for visualization
Xf, Yf, Zf = np.meshgrid(np.linspace(-2, 2, N_field), np.linspace(-2, 2, N_field), np.linspace(-2, 2, N_field))
Xf, Yf, Zf = Xf.flatten(), Yf.flatten(), Zf.flatten()
Bxf, Byf, Bzf = np.zeros_like(Xf), np.zeros_like(Yf), np.zeros_like(Zf)
Bm_f = np.zeros_like(Xf)

# Create traces for visualization
cone_trace = go.Cone(
    x=Xf, y=Yf, z=Zf, u=Bxf, v=Byf, w=Bzf,
    colorscale="Viridis", colorbar_title="|B|", cmin=np.min(Bm_f), cmax=np.max(Bm_f),
    sizemode="absolute", sizeref=0.25
)
'''

# Current loop trace
theta_circle = np.linspace(0, 2 * np.pi, 100)
ring_trace_1 = go.Scatter3d(
    x=R_1 * np.cos(theta_circle), 
    y=R_1 * np.sin(theta_circle), 
    z=np.zeros_like(theta_circle),
    mode="lines",
    line=dict(color="black", width=20),
    name="Current Loop"
)

ring_trace_2 = go.Scatter3d(
    x=R_2 * np.cos(theta_circle), 
    y=R_2 * np.sin(theta_circle), 
    z=-30*np.ones_like(theta_circle),
    mode="lines",
    line=dict(color="black", width=20),
    name="Current Loop"
)

# Electron path traces
electron_traces = []
for i in range(N_beam):
    path = np.array(x_t)[:, i, :]
    electron_traces.append(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2],
        mode="lines", line=dict(color="red", width=10),
        showlegend=False
    ))

for i in range(N_beam):
    path2 = np.array(x2_t)[:, i, :]
    electron_traces.append(go.Scatter3d(
        x=path2[:, 0], y=path2[:, 1], z=path2[:, 2],
        mode="lines", line=dict(color="blue", width=10),
        showlegend=False
    ))

# Layout and Figure setup
fig = go.Figure(data=[ring_trace_1] + [ring_trace_2] + electron_traces)  # Add electron paths to the figure
fig.update_layout(
    title="Electron Microscope",
    scene=dict(
        xaxis=dict(title="X", range=[-12, 12]),
        yaxis=dict(title="Y", range=[-12, 12]),
        zaxis=dict(title="Z", range=[-330, 10]),
        aspectmode="cube"  # Ensures the 3D plot has equal axis lengths
    )
)

# Display the plot
fig.show()
