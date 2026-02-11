import argparse
import os

# for directory handling
import sys
import time

import dolfin as df
import numpy as np

# ------------------------------------------------------------------------------
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
df.parameters["form_compiler"]["representation"] = "uflacs"

comm = df.MPI.comm_world
rank = df.MPI.rank(comm)
size = df.MPI.size(comm)

# Physical and numerical constants
rho = 1270.0  # Density [kg/m³]
nu = 1.49  # Kinematic viscosity [m²/s]
g = 9.81  # Gravity [m/s²]
f = 1.0  # Forcing frequency [Hz]
u_max = 0.05  # Maximal wall displacement [m]

# Time-stepping
dt_value = 1.0e-2  # Time step size
theta_value = 0.5  # Crank-Nicolson theta scheme
Tmax = 6.0  # Max simulation time

# Domain and mesh
x_min, x_max = 0.0, 0.5
z_min, z_max = 0.0, 1.0
x_div, z_div = 50, 100


# ------------------------------------------------------------------------------
# MESH AND BOUNDARY DEFINITIONS
# ------------------------------------------------------------------------------


def build_mesh():
    """Create rectangular mesh and mark boundaries."""
    mesh = df.RectangleMesh(
        df.Point(x_min, z_min), df.Point(x_max, z_max), x_div, z_div, "crossed"
    )
    boundary_parts = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    # unit outer domain normal vector
    norm = df.FacetNormal(mesh)
    x = df.SpatialCoordinate(mesh)

    # Boundary markers
    class Bottom(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], z_min)

    class Top(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], z_max)

    class Left(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], x_min)

    class Right(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], x_max)

    Bottom().mark(boundary_parts, 1)
    Top().mark(boundary_parts, 2)
    Left().mark(boundary_parts, 3)
    Right().mark(boundary_parts, 4)

    # Surface measure
    ds = df.Measure("ds")(subdomain_data=boundary_parts)

    class Omega(df.SubDomain):
        def inside(self, x, on_boundary):
            return True

    omega = Omega()

    # Find top-right vertex index for probe output
    tr_index = None
    for v in df.vertices(mesh):
        if df.near(v.point().x(), x_max) and df.near(v.point().y(), z_max):
            tr_index = v.index()
            break

    return mesh, boundary_parts, omega, tr_index, ds, norm


# ------------------------------------------------------------------------------
# FUNCTION SPACES
# ------------------------------------------------------------------------------


def build_spaces(mesh):
    """Create scalar, vector, and mixed function spaces (V, P, and free-surface V)."""
    P_ele = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    V_ele = df.VectorElement("CG", mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, V_ele)
    P = df.FunctionSpace(mesh, P_ele)
    # Mixed: velocity (V), pressure (P), and free-surface displacement (V)
    M = df.FunctionSpace(mesh, df.MixedElement([V_ele, P_ele, V_ele]))
    return V, P, M


# ------------------------------------------------------------------------------
# VARIATIONAL FORMS
# ------------------------------------------------------------------------------


def build_forms(M, V, dt, ds, norm, theta):
    """
    Build coupled variational form for Navier–Stokes + free surface.
    """
    # Test and trial functions
    w_ = df.TestFunction(M)
    v_, p_, h_ = df.split(w_)
    w = df.Function(M)
    v, p, h = df.split(w)
    w_k = df.Function(M)
    v_k, p_k, h_k = df.split(w_k)

    # Auxiliary functions
    dh = df.Function(V)

    # Basic subforms
    def a(v, u_):
        D = df.sym(df.grad(v))
        return (
            rho * df.inner(df.dot(df.grad(v), (v - (h - h_k) / dt)), u_)
            + df.inner(2 * nu * D, df.grad(u_))
        ) * df.dx

    def b(q, v_):
        return df.inner(df.div(v_), q) * df.dx

    def c(v_):
        return -rho * g * v_[1] * df.dx

    # Navier–Stokes residuals
    F0 = a(v_k, v_) - b(p, v_) - c(v_) + b(p_, v)
    F1 = a(v, v_) - b(p, v_) - c(v_) + b(p_, v)
    F_NS = rho * df.inner((v - v_k), v_) / dt * df.dx + (1.0 - theta) * F0 + theta * F1

    # Free-surface form
    # for clarity the form is split into three terms
    term_laplace = (
        df.inner(df.nabla_grad(h), df.nabla_grad(h_)) * df.dx
    )  # mesh smoothing
    term_symmetry = (
        -df.inner(df.nabla_grad(h[1]), norm) * h_[1] * ds(2)
    )  # consistency term

    kinematic_residual = (
        h[1]
        - h_k[1]
        + dt * (v[0] * (norm[1] * h[1].dx(0) - norm[0] * h[1].dx(1)) / norm[1] - v[1])
    )
    gamma_h = df.Constant(0.005 / x_div)  # penalty parameter
    nitsche_weight = (
        df.inner(df.nabla_grad(h_[1]), norm) - h_[1] / gamma_h
    )  # nitsche penalty weight
    term_nitsche = -kinematic_residual * nitsche_weight * ds(2)  # the full term

    # the full free-surface form
    F_h = term_laplace + term_symmetry + term_nitsche

    # Combined monolithic form
    F = F_NS + F_h
    J = df.derivative(F, w)
    return F, J, w, w_k, h, h_k, dh


# ------------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# ------------------------------------------------------------------------------

# Displacement of the walls
disp_left_expr = df.Expression(
    ("u_max*sin(2*pi*f*t)", "0"), u_max=u_max, f=f, t=0.0, degree=2
)
disp_right_expr = df.Expression(
    ("-u_max*sin(2*pi*f*t)", "0"), u_max=u_max, f=f, t=0.0, degree=2
)

# Velocity of the walls
vel_left_expr = df.Expression(
    ("2*pi*f*u_max*cos(2*pi*f*t)", "0"), u_max=u_max, f=f, t=0.0, degree=2
)
vel_right_expr = df.Expression(
    ("-2*pi*f*u_max*cos(2*pi*f*t)", "0"), u_max=u_max, f=f, t=0.0, degree=2
)


def build_bcs(
    M, boundaries, disp_left_expr, disp_right_expr, vel_left_expr, vel_right_expr
):
    """Free-slip velocity + fixed free-surface BCs."""
    # Velocity BCs
    bc_v_bot = df.DirichletBC(M.sub(0).sub(1), df.Constant(0.0), boundaries, 1)
    bc_v_left = df.DirichletBC(M.sub(0), vel_left_expr, boundaries, 3)
    bc_v_right = df.DirichletBC(M.sub(0), vel_right_expr, boundaries, 4)

    # Free-surface BCs
    bc_h_bot = df.DirichletBC(M.sub(2).sub(1), df.Constant(0.0), boundaries, 1)
    bc_h_left = df.DirichletBC(M.sub(2), disp_left_expr, boundaries, 3)
    bc_h_right = df.DirichletBC(M.sub(2), disp_right_expr, boundaries, 4)

    return [bc_v_bot, bc_v_left, bc_v_right, bc_h_bot, bc_h_left, bc_h_right]


# ------------------------------------------------------------------------------
# SOLVER CREATION
# ------------------------------------------------------------------------------


def make_solver(F, w, bcs, J):
    """Create nonlinear Newton solver with MUMPS backend."""
    problem = df.NonlinearVariationalProblem(F, w, bcs, J)
    solver = df.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = "newton"
    prm["newton_solver"]["absolute_tolerance"] = 1e-8
    prm["newton_solver"]["relative_tolerance"] = 1e-8
    prm["newton_solver"]["maximum_iterations"] = 20
    prm["newton_solver"]["linear_solver"] = "mumps"
    return solver


# ------------------------------------------------------------------------------
# OUTPUT UTILITIES
# ------------------------------------------------------------------------------


def make_xdmf_writers(comm, outdir):
    """Prepare XDMF writers for v, p, and h."""
    # make sure the directory exists
    if comm.Get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
    comm.Barrier()  # Wait for rank 0 to create dir

    file_v = df.XDMFFile(comm, os.path.join(outdir, "v.xdmf"))
    file_p = df.XDMFFile(comm, os.path.join(outdir, "p.xdmf"))
    file_h = df.XDMFFile(comm, os.path.join(outdir, "h.xdmf"))

    for f in [file_v, file_p, file_h]:
        f.parameters["flush_output"] = True
        f.parameters["rewrite_function_mesh"] = True
    return file_v, file_p, file_h


def write_outputs(file_v, file_p, file_h, w, t):
    """Write velocity, pressure, and free-surface fields."""
    v, p, h = w.split(True)
    v.rename("v", "v")
    p.rename("p", "p")
    h.rename("h", "h")
    file_v.write(v, t)
    file_p.write(p, t)
    file_h.write(h, t)


# HACK: when using MPI, only one processor has the vertex
# if the processor does not have the vertex, it supposes it has it and assigns its y coord to be very small
# the true y coord is the maximum of all y coords across the processors


def append_probe(mesh, tr_index, t, outdir):
    """Log displacement of top-right vertex."""

    local_y = -1.0e9
    if tr_index is not None:
        local_y = df.Vertex(mesh, tr_index).point().y()
    global_y = df.MPI.max(comm, local_y)

    # tell only rank 0 to write to the file
    if rank == 0:
        path = os.path.join(outdir, "topo_right_top.dat")
        with open(path, "a") as f:
            f.write(f"{t:<15.6f} {global_y - z_max}\n")


# ------------------------------------------------------------------------------
# MAIN TIME LOOP
# ------------------------------------------------------------------------------


def main(outdir):
    """Main program driver."""
    mesh, boundaries, omega, tr_index, ds, norm = build_mesh()
    V, P, M = build_spaces(mesh)
    dt = df.Constant(dt_value)

    theta = theta_value

    # Force and variational setup
    F, J, w, w_k, h, h_k, dh = build_forms(M, V, dt, ds, norm, theta)
    bcs = build_bcs(
        M, boundaries, disp_left_expr, disp_right_expr, vel_left_expr, vel_right_expr
    )
    solver = make_solver(F, w, bcs, J)

    # Initial conditions
    w_init = df.Expression(("0.0", "0.0", "0.0", "0.0", "0.0"), degree=1)
    w.interpolate(w_init)
    w_k.interpolate(w_init)

    # Initial displacement and velocity of the walls
    disp_left_expr.t = 0.0
    disp_right_expr.t = 0.0
    vel_left_expr.t = 0.0
    vel_right_expr.t = 0.0

    # Output setup
    file_v, file_p, file_h = make_xdmf_writers(comm, outdir)
    write_outputs(file_v, file_p, file_h, w, 0.0)

    # Time integration
    t = 0.0
    step = 0
    print(f"dt = {dt_value}, theta = {theta_value}, Tmax = {Tmax}")
    while t < Tmax:
        t += dt_value
        step += 1
        time_start = time.time()

        if rank == 0:
            print(f"Step {step:04d} | t = {t:.4f}")

        # Update the BC expressions
        disp_left_expr.t = t
        disp_right_expr.t = t
        vel_left_expr.t = t
        vel_right_expr.t = t

        # Solve coupled problem
        solver.solve()

        # Mesh motion: ALE update
        dh.assign(df.project(h - h_k, V))
        df.ALE.move(mesh, dh)

        # Output
        write_outputs(file_v, file_p, file_h, w, t)
        append_probe(mesh, tr_index, t, outdir)

        # Advance
        w_k.assign(w)

        time_end = time.time()
        if rank == 0:
            print("time spent in one time step = ", time_end - time_start)

    if rank == 0:
        print("Simulation complete.")


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory for output"
    )
    args = parser.parse_args()

    main(args.outdir)
