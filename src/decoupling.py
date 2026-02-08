import os
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
phi_max = 4.0 * np.pi / 180.0  # Max oscillation angle [rad]

# Time-stepping
dt_value = 1.0e-2  # Time step size
theta_value = 0.5  # Crank-Nicolson theta scheme
Tmax = 6.0  # Max simulation time

# Domain and mesh
x_min, x_max = 0.0, 1.0
z_min, z_max = 0.0, 0.3
x_div, z_div = 50, 25

# Output directory
outdir = "data/decoupling"
os.makedirs(outdir, exist_ok=True)

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
    ds = df.Measure("ds")[boundary_parts]

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
    if tr_index is None:
        raise RuntimeError("Top-right vertex not found.")

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
    M_ns = df.FunctionSpace(mesh, df.MixedElement([V_ele, P_ele]))
    M_h = df.FunctionSpace(mesh, df.MixedElement([V_ele]))
    return V, P, M_ns, M_h


# ------------------------------------------------------------------------------
# TIME-DEPENDENT FORCING
# ------------------------------------------------------------------------------


def make_force_expression():
    """Time-varying gravitational forcing vector."""
    return df.Expression(
        ("rho*g*sin(phi_max*sin(2*pi*f*t))", "-rho*g*cos(phi_max*sin(2*pi*f*t))"),
        rho=rho,
        g=g,
        phi_max=phi_max,
        f=f,
        t=0.0,
        degree=1,
    )


# ------------------------------------------------------------------------------
# VARIATIONAL FORMS
# ------------------------------------------------------------------------------


def build_ns_form(M, V, dt, ds, norm, theta):
    """
    Build variational form for Navier–Stokes.
    """
    # Test and trial functions
    w_ = df.TestFunction(M)
    v_, p_ = df.split(w_)
    h_ = df.TestFunction(V)

    w = df.Function(M)
    v, p = df.split(w)
    h = df.Function(V)
    w_k = df.Function(M)
    v_k, p_k = df.split(w_k)
    h_k = df.Function(V)

    # Auxiliary and forcing fields
    force = df.Function(V)
    force_k = df.Function(V)

    # Basic subforms
    def a(v, u_):
        D = df.sym(df.grad(v))
        return (
            rho * df.inner(df.grad(v) * (v - (h - h_k) / dt), u_)
            + df.inner(2 * nu * D, df.grad(u_))
        ) * df.dx

    def b(q, v_):
        return df.inner(df.div(v_), q) * df.dx

    def c(f_, v_):
        return df.dot(f_, v_) * df.dx

    # Navier–Stokes residuals
    F0 = a(v_k, v_) - b(p, v_) - c(force_k, v_) + b(p_, v)
    F1 = a(v, v_) - b(p, v_) - c(force, v_) + b(p_, v)
    F_ns = rho * df.inner((v - v_k), v_) / dt * df.dx + (1.0 - theta) * F0 + theta * F1
    J_ns = df.derivative(F_ns, w)
    return F_ns, J_ns, w, w_k, force, force_k


def build_surf_form(M, V, dt, ds, norm, theta):
    """
    Build  variational form for free surface.
    """
    # Test and trial functions
    h_ = df.TestFunction(V)

    w = df.Function(M)
    v = df.split(w)
    h = df.Function(V)
    h_k = df.Function(V)

    # Auxiliary and forcing fields
    dh = df.Function(V)

    # Free-surface form
    gamma_h = df.Constant(0.005 / x_div)
    F_h = (
        df.inner(df.nabla_grad(h), df.nabla_grad(h_)) * df.dx
        - df.inner(df.dot(df.nabla_grad(h), norm), h_) * ds(2)
        - df.inner(
            (
                h
                - h_k
                + dt * (v * (norm[1] * h.dx(0) - norm[0] * h.dx(1)) / norm[1] - v)
            ),
            df.dot(df.nabla_grad(h_), norm) - h_ / gamma_h,
        )
        * ds(2)
    )
    J_h = df.derivative(F_h, w)

    return F_h, J_h, h, h_k, dh


# ------------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# ------------------------------------------------------------------------------


def build_ns_bcs(M, V, boundaries, omega):
    """Free-slip velocity."""
    bc_v_bot = df.DirichletBC(M.sub(0).sub(1), df.Constant(0.0), boundaries, 1)
    bc_v_left = df.DirichletBC(M.sub(0).sub(0), df.Constant(0.0), boundaries, 3)
    bc_v_right = df.DirichletBC(M.sub(0).sub(0), df.Constant(0.0), boundaries, 4)
    bcs_vp = [bc_v_bot, bc_v_left, bc_v_right]
    return bcs_vp


def build_srf_bcs(M, V, boundaries, omega):
    """Fixed free-surface BCs."""

    bc_h_bot = df.DirichletBC(M.sub(2).sub(1), df.Constant(0.0), boundaries, 1)
    bcs_hx = df.DirichletBC(M.sub(2).sub(0), df.Constant(0.0), omega)
    bcs_h = [bc_h_bot, bcs_hx]
    return bcs_h


# ------------------------------------------------------------------------------
# SOLVERS CREATION
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


def make_xdmf_writers(comm):
    """Prepare XDMF writers for v, p, and h."""
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


def append_probe(mesh, tr_index, t):
    """Log displacement of top-right vertex."""
    tr_z = df.Vertex(mesh, tr_index).point().y()
    path = os.path.join(outdir, "topo_right_top.dat")
    with open(path, "a") as f:
        f.write(f"{t:<15.6f} {tr_z - z_max}\n")


# ------------------------------------------------------------------------------
# MAIN TIME LOOP
# ------------------------------------------------------------------------------


def main():
    """Main program driver."""
    mesh, boundaries, omega, tr_index, ds, norm = build_mesh()
    V, P, M_ns, M_h = build_spaces(mesh)
    dt = df.Constant(dt_value)

    theta = theta_value

    # Force and variational setup
    force_expr = make_force_expression()
    F_ns, J_ns, w, w_k, force, force_k = build_ns_form(M_ns, V, dt, ds, norm, theta)
    F_h, J_h, h, h_k, dh = build_surf_form(M_h, V, dt, ds, norm, theta)
    bcs_ns = build_ns_bcs(M_ns, V, boundaries, omega)
    bcs_h = build_srf_bcs(M_h, V, boundaries, omega)
    solver_ns = make_solver(F_ns, w, bcs_ns, J_ns)
    solver_h = make_solver(F_h, w, bcs_h, J_h)

    # Initial conditions
    w_init = df.Expression(("0.0", "0.0", "0.0"), degree=1)
    w.interpolate(w_init)
    w_k.interpolate(w_init)

    h_init = df.Expression(("0.0", "0.0"), degree=1)
    h.interpolate(h_init)
    h_k.assign(h_init)

    force_expr.t = 0.0
    force.assign(df.project(force_expr, V))
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
        force.assign(df.project(force_expr, V))

        # Solve coupled problem
        solver_ns.solve()
        solver_h.solve()

        # Mesh motion: ALE update
        dh.assign(df.project(h - h_k, V))
        df.ALE.move(mesh, dh)

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
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
