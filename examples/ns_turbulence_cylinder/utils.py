import numpy as np
import jax.numpy as jnp


def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    v = jnp.zeros_like(y)
    return u, v

def check_values(name, values, min_val, max_val):
    too_small = (values < min_val).any()
    too_large = (values > max_val).any()

    wandb.log({
        f"{name}_min_val": min_val,
        f"{name}_max_val": max_val,
        f"{name}_too_small": too_small,
        f"{name}_too_large": too_large
    })

    if too_small:
        print(f"Values in {name} are too small")
    if too_large:
        print(f"Values in {name} are too large")

def get_dataset():
    data = jnp.load("data/ns_unsteady.npy", allow_pickle=True).item()
    print('u_ref: ', type(data["u"].astype(float)), data["u"].shape)
    u_ref = jnp.array(data["u"].astype(float))
    v_ref = jnp.array(data["v"].astype(float))
    p_ref = jnp.array(data["p"].astype(float))
    k_ref = jnp.array(data["k"].astype(float)) #me
    omega_ref = jnp.array(data["omega"].astype(float)) #me
    #t = jnp.array(data["t"])
    coords = jnp.array(data["coords"])
    inflow_coords = jnp.array(data["inflow_coords"].astype(float))
    outflow_coords = jnp.array(data["outflow_coords"].astype(float))
    symmetry_coords = jnp.array(data["symmetry_coords"].astype(float)) #me
    cylinder_coords = jnp.array(data["cylinder_coords"].astype(float))
    nu = jnp.array(data["nu"])

    return (
        u_ref,
        v_ref,
        p_ref,
        k_ref, #me
        omega_ref, #me
        coords,
        inflow_coords,
        outflow_coords,
        symmetry_coords, #me
        cylinder_coords,
        nu,
    )


def get_fine_mesh():
    data = jnp.load("data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = jnp.array(data["coords"])

    data = jnp.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = jnp.array(data["coords"])

    return fine_coords, fine_coords_near_cyl
def calculate_range_and_divergence(variable, name):
    var_min = np.min(variable)
    var_max = np.max(variable)
    var_range = var_max - var_min
    var_std = np.std(variable)

    print(f'{name}')
    print('Type: ', type(variable), '  Shape: ', variable.shape)
    print(f'{name} Range: min = {var_min}, max = {var_max}, range = {var_range}')
    print(f'{name} Divergence (Standard Deviation): {var_std}')
    print('')

