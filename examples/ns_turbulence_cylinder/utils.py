import jax.numpy as jnp


#def parabolic_inflow(y, U_max):
#    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
#    v = jnp.zeros_like(y)
#    return u, v


def get_dataset():
    data = jnp.load("data/ns_unsteady.npy", allow_pickle=True).item()
    u_ref = jnp.array(data["u"])
    v_ref = jnp.array(data["v"])
    p_ref = jnp.array(data["p"])
    k_ref = jnp.array(data["k"]) #me
    omega_ref = jnp.array(data["w"]) #me
    conc_ref = jnp.array(data["omega"]) #me
    t = jnp.array(data["t"])
    coords = jnp.array(data["coords"])
    inflow_coords = jnp.array(data["inflow_coords"])
    outflow_coords = jnp.array(data["outflow_coords"])
    symm_coords = jnp.array(data["symm_coords"]) #me
    cylinder_coords = jnp.array(data["cylinder_coords"])
    nu = jnp.array(data["nu"])

    return (
        u_ref,
        v_ref,
        p_ref,
        k_ref, #me
        omega_ref, #me
        conc_ref, #me
        coords,
        inflow_coords,
        outflow_coords,
        symm_coords, #me
        cylinder_coords,
        nu,
    )


def get_fine_mesh():
    data = jnp.load("data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = jnp.array(data["coords"])

    data = jnp.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = jnp.array(data["coords"])

    return fine_coords, fine_coords_near_cyl
