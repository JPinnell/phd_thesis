# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:36:04 2025

@author: jmcp5
"""

import numpy as np
import numpy.linalg as la
from math import pi
# Added import statements
import numba
from scipy.linalg import lu_factor, lu_solve

# -----------------------------
# 1. Mesh Generation Utilities
# -----------------------------

def sphere_mesh(center, radius, n_lat, n_lon):
    """
    Generate a simple triangulated mesh for a sphere of given 'radius' and 'center'.
    n_lat: number of latitude divisions (not counting poles).
    n_lon: number of longitude divisions.

    Returns:
        nodes (N x 3): array of mesh vertex coordinates
        faces (M x 3): array of triangular face indices
    """
    lat_vals = np.linspace(0, pi, n_lat+1)
    lon_vals = np.linspace(0, 2*pi, n_lon, endpoint=False)

    node_list = []
    # South pole
    node_list.append([0, 0, -radius])
    # Intermediate latitudes
    for i in range(1, n_lat):
        theta = lat_vals[i]
        z = -radius*np.cos(theta)
        r_xy = radius*np.sin(theta)
        for phi in lon_vals:
            x = r_xy*np.cos(phi)
            y = r_xy*np.sin(phi)
            node_list.append([x, y, z])
    # North pole
    node_list.append([0, 0, radius])
    node_list = np.array(node_list)
    node_list += center

    # Build faces
    faces = []
    south_pole_index = 0
    north_pole_index = 1 + (n_lat-1)*n_lon

    def ring_index(i, j):
        return 1 + i*n_lon + j

    # Triangles connecting south pole to first ring
    for j in range(n_lon):
        j_next = (j+1) % n_lon
        faces.append([south_pole_index,
                      ring_index(0, j_next),
                      ring_index(0, j)])

    # Triangles for intermediate latitudes
    for i in range(n_lat-2):
        for j in range(n_lon):
            j_next = (j+1) % n_lon
            v1 = ring_index(i, j)
            v2 = ring_index(i, j_next)
            v3 = ring_index(i+1, j)
            v4 = ring_index(i+1, j_next)
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    # Triangles connecting top ring to north pole
    for j in range(n_lon):
        j_next = (j+1) % n_lon
        faces.append([north_pole_index,
                      ring_index(n_lat-2, j),
                      ring_index(n_lat-2, j_next)])
    return np.array(node_list), np.array(faces, dtype=int)

def build_dumbbell_mesh(radius, separation, n_lat, n_lon):
    """
    Generate a triangulated mesh for a dumbbell particle consisting of two spheres.
    
    Parameters:
    -----------
    radius : float
        Radius of each sphere.
    separation : float
        Distance between the centers of the two spheres.
    n_lat : int
        Number of latitude subdivisions for each sphere.
    n_lon : int
        Number of longitude subdivisions for each sphere.
    
    Returns:
    --------
    nodes : ndarray of shape (2*N_nodes, 3)
        Combined array of mesh vertex coordinates for both spheres.
    faces : ndarray of shape (2*N_faces, 3)
        Combined array of triangular face indices for both spheres.
    """
    # Calculate centers of the two spheres along the x-axis
    center1 = np.array([-separation / 2, 0.0, 0.0])
    center2 = np.array([separation / 2, 0.0, 0.0])
    
    # Generate mesh for the first sphere
    nodes1, faces1 = sphere_mesh(center1, radius, n_lat=n_lat, n_lon=n_lon)
    
    # Generate mesh for the second sphere
    nodes2, faces2 = sphere_mesh(center2, radius, n_lat=n_lat, n_lon=n_lon)
    
    # Combine nodes
    nodes = np.vstack((nodes1, nodes2))
    
    # Adjust face indices for the second sphere
    faces2_adjusted = faces2 + len(nodes1)
    
    # Combine faces
    faces = np.vstack((faces1, faces2_adjusted))
    
    return nodes, faces


def build_supershiner_mesh(r1, r2, n_lat, n_lon):
    """
    Generate a triangulated mesh for a supershiner particle consisting of two spheres:
      - Outer sphere: radius r1, centered at origin (0, 0, 0)
      - Inner sphere: radius r2, centered at (r1, 0, 0)
    
    Parameters:
    -----------
    r1 : float
        Radius of the outer sphere.
    r2 : float
        Radius of the inner sphere.
    n_lat : int, optional
        Number of latitude subdivisions for each sphere. Default is 6.
    n_lon : int, optional
        Number of longitude subdivisions for each sphere. Default is 12.
    
    Returns:
    --------
    nodes : ndarray of shape (2*N_nodes, 3)
        Combined array of mesh vertex coordinates for both spheres.
    faces : ndarray of shape (2*N_faces, 3)
        Combined array of triangular face indices for both spheres.
    
    Notes:
    ------
    - This function leverages the existing `sphere_mesh` function to generate each sphere's mesh.
    - The inner sphere is positioned such that its center lies exactly on the outer sphere's surface.
    - The function ensures correct indexing by offsetting the inner sphere's face indices.
    """
    import numpy as np
    
    # Define centers for both spheres
    center_outer = np.array([0.0, 0.0, 0.0])      # Outer sphere centered at origin
    center_inner = np.array([0.0, 0.0, r1])       # Inner sphere centered at (r1, 0, 0)
    
    # Generate mesh for the outer sphere
    nodes_outer, faces_outer = sphere_mesh(center_outer, r1, n_lat=n_lat, n_lon=n_lon)
    
    # Generate mesh for the inner sphere
    nodes_inner, faces_inner = sphere_mesh(center_inner, r2, n_lat=n_lat, n_lon=n_lon)
    
    # Combine the nodes from both spheres
    nodes = np.vstack((nodes_outer, nodes_inner))
    
    # Adjust face indices for the inner sphere to account for the outer sphere's nodes
    faces_inner_adjusted = faces_inner + len(nodes_outer)
    
    # Combine the faces from both spheres into a single array
    faces = np.vstack((faces_outer, faces_inner_adjusted))
    
    return nodes, faces


# ------------------------------
# 2. Basic Utility / Kernel Functions
# ------------------------------

# Applied Recommendation 1: Added @numba.njit decorator
@numba.njit
def stokeslet_kernel(x_target, x_source, mu):
    """
    Return the 3x3 Stokeslet kernel G(x_target, x_source):
      G(r) = 1 / (8 pi mu r) [ I + r-hat (r-hat)^T ]
    where r = x_target - x_source
    """
    r_vec = x_target - x_source
    # Applied Recommendation 2: Precompute norms and reuse variables
    r_sqr = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]
    if r_sqr < 1e-14:
        return np.zeros((3,3))
    r = np.sqrt(r_sqr)
    inv_r = 1.0 / r
    inv_r3 = inv_r / r_sqr
    inv_8pi_mu = 1.0 / (8.0 * pi * mu)
    c = inv_8pi_mu * inv_r
    # Compute outer product of r_hat with itself
    rhat = r_vec * inv_r
    rhat_outer = np.outer(rhat, rhat)
    # Build G
    G = c * (np.identity(3) + rhat_outer)
    return G

def triangle_area_and_centroid(xa, xb, xc):
    """
    Return area and centroid of a triangle with vertices xa, xb, xc in 3D.
    """
    cross_prod = np.cross(xb - xa, xc - xa)
    area = 0.5*la.norm(cross_prod)
    centroid = (xa + xb + xc)/3.0
    return area, centroid

# Applied Recommendation 1: Added @numba.njit decorator
@numba.njit
def assemble_single_layer(nodes, faces, mu):
    """
    Assemble the single-layer operator L (3N x 3N), where N = number of faces.
    Each row-block i corresponds to the collocation point on face i.
    Each column-block j corresponds to the traction on face j.

    We'll:
      1. Collocate at the face centroids.
      2. Use the naive approach with centroid-based integration.
    """
    Nfaces = len(faces)
    L = np.zeros((3*Nfaces, 3*Nfaces))
    collocation_points = np.zeros((Nfaces, 3))
    areas = np.zeros(Nfaces)

    # Precompute centroids & areas
    for i in range(Nfaces):
        tri = faces[i]
        xa, xb, xc = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        # Applied Recommendation 2: Precompute variables inside loops
        cross_prod = np.cross(xb - xa, xc - xa)
        area_i = 0.5*np.linalg.norm(cross_prod)
        centroid_i = (xa + xb + xc)/3.0
        collocation_points[i] = centroid_i
        areas[i] = area_i

    for i in range(Nfaces):
        xi = collocation_points[i]
        for j in range(Nfaces):
            xj = collocation_points[j]
            Aj = areas[j]
            if i == j:
                # Skip self-interaction or handle separately if desired
                continue
            # Applied Recommendation 1: Ensure function calls are compatible with Numba
            G = stokeslet_kernel(xi, xj, mu)
            L_block = G * Aj
            L[3*i:3*i+3, 3*j:3*j+3] = L_block

    return L, collocation_points, areas

# -----------------------------------------
# 3. System Assembly, Force/Torque, Solving
# -----------------------------------------

def assemble_global_system(L, faces, colloc_pts, areas, Fext, Text, mu):
    """
    Build the linear system for:
      L f = C(U,Omega)
      plus the net force & net torque constraints:
        sum(f * area) = -Fext
        sum(r x f * area) = -Text

    We'll store unknowns as [f(3N), U(3), Omega(3)] in a single vector.
    The system is (3N + 6) x (3N + 6).
    """
    Nfaces = len(faces)
    A = np.zeros((3*Nfaces + 6, 3*Nfaces + 6))
    rhs = np.zeros(3*Nfaces + 6)

    # Top-left block: L
    A[:3*Nfaces, :3*Nfaces] = L

    # Build C matrix for rigid-body velocity at collocation points
    # velocity_i = U + Omega x (x_i)
    C = np.zeros((3*Nfaces, 6))
    for i in range(Nfaces):
        x_i = colloc_pts[i]
        # partial wrt U
        C[3*i:3*i+3, 0:3] = np.eye(3)
        # partial wrt Omega
        r = x_i  # reference = 0 for simplicity
        cross_mat = np.array([
            [0,     -r[2],  r[1]],
            [r[2],   0,    -r[0]],
            [-r[1],  r[0],  0]
        ])
        C[3*i:3*i+3, 3:6] = cross_mat

    # Put -C in top-right block so that L f - C(U,Omega) = 0
    A[:3*Nfaces, 3*Nfaces:] = -C

    # Force & torque constraints in bottom 6 rows
    # D matrix: net force = sum over faces f_i * area_i
    #           net torque= sum over faces r_i x f_i * area_i
    D = np.zeros((6, 3*Nfaces))
    for i in range(Nfaces):
        Ai = areas[i]
        x_i = colloc_pts[i]
        # Force
        D[0:3, 3*i:3*i+3] += Ai*np.eye(3)
        # Torque
        cross_mat = np.array([
            [0,      x_i[2], -x_i[1]],
            [-x_i[2], 0,     x_i[0]],
            [x_i[1], -x_i[0], 0]
        ])
        D[3:6, 3*i:3*i+3] += Ai*cross_mat

    A[3*Nfaces:, :3*Nfaces] = D

    # The net force & torque eqn: D f = c
    # c = -[Fext, Text]
    rhs[3*Nfaces:] = np.concatenate([-Fext, -Text])

    return A, rhs

def solve_stokes_bem_lu(lu_and_piv, rhs):
    """
    Solve linear system A x = rhs using precomputed LU factorization.
    x = [f(3N), U(3), Omega(3)].
    """
    x_sol = lu_solve(lu_and_piv, rhs)
    return x_sol

# -------------------------------------------
# 4. Mobility Tensor Assembly (Apply 6 Loads)
# -------------------------------------------

def compute_mobility_tensor(nodes, faces, mu):
    """
    Compute the 6x6 mobility tensor by applying unit forces/torques in the 6 principal directions.
    """
    # 1. Assemble single-layer operator L
    L, colloc_pts, areas = assemble_single_layer(nodes, faces, mu)
    Nfaces = len(faces)

    # Build the global system matrix once
    # Since L and the constraints matrices (except RHS) are constant, we can factor A once
    # We'll use a placeholder Fext and Text to assemble A
    Fext_placeholder = np.zeros(3)
    Text_placeholder = np.zeros(3)
    A, _ = assemble_global_system(L, faces, colloc_pts, areas, Fext_placeholder, Text_placeholder, mu)

    # Applied Recommendation 3: Precompute LU factorization
    lu_and_piv = lu_factor(A)

    # We'll build the final M as 6x6
    M = np.zeros((6,6))

    # define 6 load cases:
    load_cases = []
    for i in range(3):
        F = np.zeros(3); F[i] = 1.0
        T = np.zeros(3)
        load_cases.append((F, T))
    for i in range(3):
        F = np.zeros(3)
        T = np.zeros(3); T[i] = 1.0
        load_cases.append((F, T))

    # Solve each case using the precomputed LU factorization
    for col_idx, (Fext, Text) in enumerate(load_cases):
        # Build the RHS for this load case
        rhs = np.zeros(3*Nfaces + 6)
        rhs[3*Nfaces:] = np.concatenate([Fext, Text])
        # Solve using LU factorization
        x_sol = solve_stokes_bem_lu(lu_and_piv, rhs)
        # x_sol: [f(3N), U(3), Omega(3)]
        U_sol = x_sol[3*Nfaces : 3*Nfaces+3]
        Omega_sol = x_sol[3*Nfaces+3 : 3*Nfaces+6]
        M[0:3, col_idx] = U_sol
        M[3:6, col_idx] = Omega_sol

    return M


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_mesh(nodes, faces, save=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for face in faces:
        tri = nodes[face]
        tri = np.vstack((tri, tri[0]))  # Close the triangle
        ax.plot(tri[:,0], tri[:,1], tri[:,2], color='r')
    ax.set_xlabel(r'$x/r$', fontsize=20)
    ax.set_ylabel(r'$y/r$', fontsize=20)
    ax.set_zlabel(r'$z/r$', fontsize=20)
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-1,0,1])
    plt.tight_layout()
    if save:
        plt.savefig('Mobility_mesh.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------
# 5. Main Script
# ---------------
if __name__ == "__main__":
    # Example: Single Sphere Test
    mu = 1.0                 # viscosity
    radius = 1.0

    # Mesh resolution
    n_lat = 32
    n_lon = 32

    # Build spherical particle mesh
    # center = np.array([0.0, 0.0, 0.0])
    # nodes, faces = sphere_mesh(center, radius, n_lat=n_lat, n_lon=n_lon)
    
    # Build dumbbell particle mesh
    # separation = 2.0
    # nodes, faces = build_dumbbell_mesh(radius, separation, n_lat=n_lat, n_lon=n_lon)
    
    # Build supershiner mesh
    aspect_ratio = np.linspace(1,10,10)
    m_R = np.zeros_like(aspect_ratio)
    m_T = np.zeros_like(aspect_ratio)
    m_RT = np.zeros_like(aspect_ratio)
    r1 = 1.0
    for i in range(len(aspect_ratio)):
        r2 = r1/aspect_ratio[i]
        nodes, faces = build_supershiner_mesh(r1, r2, n_lat, n_lon)

        # Compute mobility using the modified BEM code
        M = compute_mobility_tensor(nodes, faces, mu)
        
        m_RT[i] = M[0,4]
        m_T[i] = M[0,0]
        m_R[i] = M[3,3]

    
    # b_t = 1/(6*np.pi*mu*radius)
    # b_r = 1/(8*np.pi*mu*radius**3)
    # print(b_t, b_r)

    # Print result
    # np.set_printoptions(precision=5, suppress=True)
    # print("Mobility Tensor (6x6):")
    # print(M)

#%%

import matplotlib.pyplot as plt
import matplotlib

x = aspect_ratio
# y = [m_T/(1/(6*np.pi))/0.985, m_R/(1/(8*np.pi))/0.96]
y = [m_RT/(1/(8*np.pi))]

# Preamble: Customizable Parameters
params = {
    "title": "",
    "xlabel": "Aspect ratio",
    "ylabel": r"$m^{RT}/m^R$",
    "xlim": (None, None), # (None, None)
    "ylim": (None, None),
    "line_width": 3.0,
    "line_color": ["red"],
    "line_style": "-",
    "marker": "o",
    "marker_color": "black",
    "marker_size": 8,
    "grid": True,
    "grid_style": "dashed",
    "grid_color": "gray",
    "grid_alpha": 1,
    "font_size": 14,
    "font_name": "Computer Modern",
    "title_font_size": 16,
    "xlabel_font_size": 20,
    "ylabel_font_size": 22,
    "legend": False,
    "legend_text": [r"$M^{TT}_{1,1}/m^T$", r"$M^{RR}_{3,3}/m^R$"],
    "legend_loc": "best",
    "legend_font_size": 18,
    "figure_size": (8, 6),
    "dpi": 300,
    "save_file": True,
    "file_path": "C:\\Users\\Jonathan\\Desktop\\PhD\\Write up\\Simulations\\Figures\\RTcoupling.png"
}

# Apply font settings globally
matplotlib.rc('font', size=params["font_size"], family="serif")
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'

# Create the plot
plt.figure(figsize=params["figure_size"], dpi=params["dpi"])
for i in range(len(y)):
    plt.plot(
        x, y[i],
        linewidth=params["line_width"],
        color=params["line_color"][i],
        linestyle=params["line_style"],
        marker=params["marker"],
        markerfacecolor=params["marker_color"],
        markersize=params["marker_size"],
        label=params["legend_text"][i]
    )


# Customize title and labels
plt.title(params["title"], fontsize=params["title_font_size"])
plt.xlabel(params["xlabel"], fontsize=params["xlabel_font_size"])
plt.ylabel(params["ylabel"], fontsize=params["ylabel_font_size"])

# Set axis limits
plt.xlim(params["xlim"])
plt.ylim(params["ylim"])

# Customize grid
if params["grid"]:
    plt.grid(
        linestyle=params["grid_style"],
        color=params["grid_color"],
        alpha=params["grid_alpha"]
    )

# Add legend if enabled
if params["legend"]:
    plt.legend(loc=params["legend_loc"], fontsize=params["legend_font_size"])
    
if params["save_file"]:
    plt.savefig(params["file_path"], dpi=params["dpi"], bbox_inches='tight')
    
# Show the plot
plt.show()