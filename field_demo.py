import numpy as np
import plotly.graph_objects as go

# Physical Constants
MU_0 = 1  
R_1 = 1
I_1 = 15
N = 100  
N_field = 20

# Compute segment properties
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
dL_1 = 2 * R_1 * np.sin(np.pi / N)

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


def signed_log(arr):
    """
    Computes the logarithm of the absolute value of each element in the array,
    preserving the original sign.
    """
    return np.sign(arr) * np.log(np.abs(arr))

# Set up 2 lenses
# Discretized positions of the current elements
R_ring_1 = np.column_stack((R_1 * np.cos(theta), R_1 * np.sin(theta), np.zeros(N)))
dL_ring_1 = dL_1 * np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(N)))

# Prepare data for visualization
Xf, Yf, Zf = np.meshgrid(np.linspace(-2, 2, N_field), 0.1, np.linspace(-2, 2, N_field))
Xf, Yf, Zf = Xf.flatten(), Yf.flatten(), Zf.flatten()
B_field = ring_current(np.column_stack((Xf, Yf, Zf)), R_ring_1, dL_ring_1, I_1, R_1)
Bxf, Byf, Bzf = B_field[:, 0], B_field[:, 1], B_field[:, 2]
Bm_f = np.linalg.norm(B_field, axis=1)

#

# Create traces for visualization
cone_trace = go.Cone(
    x=Xf, y=Yf, z=Zf, u=Bxf/np.sqrt(B_field[:, 0]**2+B_field[:, 1]**2+B_field[:, 2]**2), v=Byf/np.sqrt(B_field[:, 0]**2+B_field[:, 1]**2+B_field[:, 2]**2), w=Bzf/np.sqrt(B_field[:, 0]**2+B_field[:, 1]**2+B_field[:, 2]**2),
    colorscale="Reds", colorbar_title="|B|", cmin=1, cmax=1,
    sizemode="absolute", sizeref=0.5
)

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

# Layout and Figure setup
fig = go.Figure(data=[ring_trace_1, cone_trace])  # Add electron paths to the figure
fig.update_layout(
    title="Electron Microscope",
    scene=dict(
        xaxis=dict(title="X", range=[-2, 2]),
        yaxis=dict(title="Y", range=[-2, 2]),
        zaxis=dict(title="Z", range=[-2, 2]),
        aspectmode="cube"  # Ensures the 3D plot has equal axis lengths
    )
)

# Display the plot
fig.show()
