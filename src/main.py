import os
import time

import numpy as np
from dolfin import *

# ------------------------------------------------------------------------------
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
parameters["form_compiler"]["representation"] = "uflacs"

comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)

# Physical and numerical constants
rho = 1270.0  # Density [kg/m³]
nu = 1.49  # Kinematic viscosity [m²/s]
g = 9.81  # Gravity [m/s²]
f = 1.0  # Forcing frequency [Hz]
phi_max = 4.0 * np.pi / 180.0  # Max oscillation angle [rad]

# Time-stepping
dt_value = 5.0e-3  # Time step size
theta_value = 0.5  # Crank-Nicolson theta scheme
Tmax = 6.0  # Max simulation time

# Domain and mesh
x_min, x_max = 0.0, 1.0
z_min, z_max = 0.0, 0.3
x_div, z_div = 50, 25

# Output directory
outdir = "data-out-nitsche"
os.makedirs(outdir, exist_ok=True)


# ------------------------------------------------------------------------------
# MESH AND BOUNDARY DEFINITIONS
# ------------------------------------------------------------------------------


def build_mesh():
    """Create rectangular mesh and mark boundaries."""
    mesh = RectangleMesh(
        Point(x_min, z_min), Point(x_max, z_max), x_div, z_div, "crossed"
    )
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    # unit outer domain normal vector
    norm = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    # Boundary markers
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], z_min)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], z_max)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], x_min)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], x_max)

    Bottom().mark(boundary_parts, 1)
    Top().mark(boundary_parts, 2)
    Left().mark(boundary_parts, 3)
    Right().mark(boundary_parts, 4)

    # Surface measure
    ds = Measure("ds")[boundary_parts]

    class Omega(SubDomain):
        def inside(self, x, on_boundary):
            return True

    omega = Omega()

    # Find top-right vertex index for probe output
    tr_index = None
    for v in vertices(mesh):
        if near(v.point().x(), x_max) and near(v.point().y(), z_max):
            tr_index = v.index()
            break
    if tr_index is None:
        raise RuntimeError("Top-right vertex not found.")

    return mesh, boundary_parts, omega, tr_index, ds, norm


# ------------------------------------------------------------------------------
# FUNCTION SPACES
# ------------------------------------------------------------------------------


def build_spaces(mesh):
    """Create scalar, vector, and mixed function spaces (V, P, and free-surface V)."""
    P_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, V_ele)
    P = FunctionSpace(mesh, P_ele)
    # Mixed: velocity (V), pressure (P), and free-surface displacement (V)
    M = FunctionSpace(mesh, MixedElement([V_ele, P_ele, V_ele]))
    return V, P, M


# ------------------------------------------------------------------------------
# TIME-DEPENDENT FORCING
# ------------------------------------------------------------------------------


def make_force_expression():
    """Time-varying gravitational forcing vector."""
    return Expression(
        ("0.0", "-rho*g"),
        rho=rho,
        g=g,
        t=0.0,
        degree=1,
    )


def make_wall_position_expression():
    """Motion of the left wall."""
    return Expression(
        ("u0*sin(2*pi*f*t)", "0.0"),
        u0=0.05,
        f=1,
        t=0.0,
        degree=1,
    )


def make_wall_velocity_expression():
    """Velocity of the left wall."""
    return Expression(
        ("2*pi*f*u0*cos(2*pi*f*t)", "0.0"),
        u0=0.05,
        f=1,
        t=0.0,
        degree=1,
    )


# ------------------------------------------------------------------------------
# VARIATIONAL FORMS
# ------------------------------------------------------------------------------


def build_forms(M, V, dt, ds, norm, theta):
    """
    Build coupled variational form for Navier–Stokes + free surface.
    """
    # Test and trial functions
    w_ = TestFunction(M)
    v_, p_, h_ = split(w_)
    w = Function(M)
    v, p, h = split(w)
    w_k = Function(M)
    v_k, p_k, h_k = split(w_k)

    # Auxiliary and forcing fields
    dh = Function(V)
    force = Function(V)
    force_k = Function(V)

    # Basic subforms
    def a(v, u_):
        D = sym(grad(v))
        return (
            rho * inner(grad(v) * (v - (h - h_k) / dt), u_)
            + inner(2 * nu * D, grad(u_))
        ) * dx

    def b(q, v_):
        return inner(div(v_), q) * dx

    def c(f_, v_):
        return dot(f_, v_) * dx

    # Navier–Stokes residuals
    F0 = a(v_k, v_) - b(p, v_) - c(force_k, v_) + b(p_, v)
    F1 = a(v, v_) - b(p, v_) - c(force, v_) + b(p_, v)
    F_NS = rho * inner((v - v_k), v_) / dt * dx + (1.0 - theta) * F0 + theta * F1

    # Free-surface form
    gamma_h = Constant(0.005 / x_div)
    F_h = (
        inner(nabla_grad(h), nabla_grad(h_)) * dx
        - inner(nabla_grad(h[1]), norm) * h_[1] * ds(2)
        - (
            h[1]
            - h_k[1]
            + dt
            * (v[0] * (norm[1] * h[1].dx(0) - norm[0] * h[1].dx(1)) / norm[1] - v[1])
        )
        * (inner(nabla_grad(h_[1]), norm) - h_[1] / gamma_h)
        * ds(2)
    )

    # Combined monolithic form
    F = F_NS + F_h
    J = derivative(F, w)
    return F, J, w, w_k, force, force_k, h, h_k, dh


# ------------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# ------------------------------------------------------------------------------


def build_bcs(M, V, boundaries, omega, wall_vel_expr):
    """Free-slip velocity + fixed free-surface BCs."""
    # Velocity BCs
    bc_v_bot = DirichletBC(M.sub(0).sub(1), Constant(0.0), boundaries, 1)
    bc_v_left = DirichletBC(M.sub(0), wall_vel_expr, boundaries, 3)
    # bc_v_left = DirichletBC(M.sub(0).sub(0), Constant(0.0), boundaries, 3)
    bc_v_right = DirichletBC(M.sub(0).sub(0), Constant(0.0), boundaries, 4)
    bcs_vp = [bc_v_bot, bc_v_left, bc_v_right]

    # Free-surface BCs (Dirichlet on all sides)
    bc_h_bot = DirichletBC(M.sub(2).sub(1), Constant(0.0), boundaries, 1)
    bcs_hx = DirichletBC(M.sub(2).sub(0), Constant(0.0), omega)
    bcs_h = [bc_h_bot, bcs_hx]

    return bcs_vp + bcs_h


# ------------------------------------------------------------------------------
# SOLVER CREATION
# ------------------------------------------------------------------------------


def make_solver(F, w, bcs, J):
    """Create nonlinear Newton solver with MUMPS backend."""
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)
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


def make_xdmf_writers(comm):
    """Prepare XDMF writers for v, p, and h."""
    file_v = XDMFFile(comm, os.path.join(outdir, "v.xdmf"))
    file_p = XDMFFile(comm, os.path.join(outdir, "p.xdmf"))
    file_h = XDMFFile(comm, os.path.join(outdir, "h.xdmf"))
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


def append_probe(mesh, tr_index, t):
    """Log displacement of top-right vertex."""
    tr_z = Vertex(mesh, tr_index).point().y()
    path = os.path.join(outdir, "topo_right_top.dat")
    with open(path, "a") as f:
        f.write(f"{t:<15.6f} {tr_z - z_max}\n")


# ------------------------------------------------------------------------------
# MAIN TIME LOOP
# ------------------------------------------------------------------------------


def main():
    """Main program driver."""
    mesh, boundaries, omega, tr_index, ds, norm = build_mesh()
    V, P, M = build_spaces(mesh)
    dt = Constant(dt_value)

    theta = theta_value

    # Force and variational setup
    force_expr = make_force_expression()
    wall_vel_expr = make_wall_velocity_expression()
    F, J, w, w_k, force, force_k, h, h_k, dh = build_forms(M, V, dt, ds, norm, theta)
    bcs = build_bcs(M, V, boundaries, omega, wall_vel_expr)
    solver = make_solver(F, w, bcs, J)

    # Initial conditions
    w_init = Expression(("0.0", "0.0", "0.0", "0.0", "0.0"), degree=1)
    w.interpolate(w_init)
    w_k.interpolate(w_init)
    force_expr.t = 0.0
    wall_vel_expr.t = 0.0
    force.assign(project(force_expr, V))
    force_k.assign(force)

    # Output setup
    file_v, file_p, file_h = make_xdmf_writers(comm)
    write_outputs(file_v, file_p, file_h, w, 0.0)

    # Time integration
    t = 0.0
    step = 0
    print(f"dt = {dt_value}, theta = {theta_value}, Tmax = {Tmax}")
    while t < Tmax:
        time_start = time.time()
        if rank == 0:
            print(f"Step {step:04d} | t = {t:.4f}")

        # Update forcing and displacement
        force_expr.t = t
        force.assign(project(force_expr, V))
        wall_vel_expr.t = t

        # Solve coupled problem
        solver.solve()

        # Mesh motion: ALE update
        dh.assign(project(h - h_k, V))
        ALE.move(mesh, dh)

        # Output
        write_outputs(file_v, file_p, file_h, w, t)
        append_probe(mesh, tr_index, t)

        # Advance
        w_k.assign(w)
        force_k.assign(force)
        t += dt_value
        step += 1

        time_end = time.time()
        if rank == 0:
            print("time spent in one time step = ", time_end - time_start)

    if rank == 0:
        print("Simulation complete.")


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
