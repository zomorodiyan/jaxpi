from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map

import optax

from jaxpi import archs
from jaxpi.models import ForwardBVP, ForwardIVP
from jaxpi.utils import ntk_fn
from jaxpi.evaluator import BaseEvaluator


class NavierStokes2D(ForwardIVP):
    def __init__(self, config, inflow_fn, temporal_dom, coords, Re):
        print('NavierStokes2D init')
        super().__init__(config)

        self.inflow_fn = inflow_fn
        self.temporal_dom = temporal_dom
        self.coords = coords
        self.Re = Re  # Reynolds number

        # Non-dimensionalized domain length and width
        self.L, self.W = self.coords.max(axis=0) - self.coords.min(axis=0)

        if config.nondim == True:
            self.U_star = 9.0 #me
            self.L_star = 80.0 #me
        else:
          print('no implemented 15892')
          return(-1)

        # Predict functions over batch
        self.u0_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        self.v0_pred_fn = vmap(self.v_net, (None, None, 0, 0))
        self.p0_pred_fn = vmap(self.p_net, (None, None, 0, 0))
        self.k0_pred_fn = vmap(self.k_net, (None, None, 0, 0))
        self.omega0_pred_fn = vmap(self.omega_net, (None, None, 0, 0))

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0, 0))
        self.k_pred_fn = vmap(self.k_net, (None, 0, 0, 0))
        self.omega_pred_fn = vmap(self.omega_net, (None, 0, 0, 0))
        self.w_pred_fn = vmap(self.w_net, (None, 0, 0, 0)) #unused
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.temporal_dom[1]  # rescale t into [0, 1]
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        inputs = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, inputs)

        # Start with an initial state of a free stream (constant velocity) flow
        u_const = self.U_star
        u = outputs[0] + self.U_star
        v = outputs[1]
        p = outputs[2]
        k = outputs[2]
        omega = outputs[2]
        return u, v, p, k, omega

    def u_net(self, params, t, x, y):
        u, _, _, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _, _, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p, _, _ = self.neural_net(params, t, x, y)
        return p

    def k_net(self, params, t, x, y):
        _, _, _, k, _ = self.neural_net(params, t, x, y)
        return k

    def omega_net(self, params, t, x, y):
        _, _, _, _, omega = self.neural_net(params, t, x, y)
        return omega

    def w_net(self, params, t, x, y):
        u, v, _ = self.neural_net(params, t, x, y)
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        w = v_x - u_y
        return w

    def r_net(self, params, t, x, y):
        u, v, p, k, omega = self.neural_net(params, t, x, y)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        v_t = grad(self.v_net, argnums=1)(params, t, x, y)
        k_t = grad(self.k_net, argnums=1)(params, t, x, y)
        omega_t = grad(self.omega_net, argnums=1)(params, t, x, y)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        k_x = grad(self.k_net, argnums=2)(params, t, x, y)
        omega_x = grad(self.omega_net, argnums=2)(params, t, x, y)

        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)
        k_y = grad(self.k_net, argnums=3)(params, t, x, y)
        omega_y = grad(self.omega_net, argnums=3)(params, t, x, y)

        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        u_xy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y)

        v_yx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y)
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y)

        # SST k-omega constants
        a1 = 0.31
        kappa = 0.41
        alpha_0 = 1/9
        alpha_star_infinity= 1.0
        beta_i1 = 0.075
        beta_i2 = 0.0828
        beta_star_infinity = 0.09
        sigma_omega1 = 2.0
        R_beta = 8
        sigma_omega2 = 1.168
        R_k = 6
        sigma_k1 = 1.176
        R_omega = 2.95
        sigma_k2 = 1.0
        xi_star = 1.5
        M_t0 = 0.25
        f_beta = 1.0
        f_star_beta = 1.0
        #physical constants
        rho = 1.225  # kg/m^3
        mu = 1.7894e-5  # kg/(m*s)
        gamma = 1.4
        R = 287  # m^2/(s^2*K)
        T = 297.15  # K
        U_star = 9.0 #[m/s] velocity
        L_star = 80.0 #[m] diameter
        Re = rho * U_star * L_star / mu
        # function to calcualte y_hat which is distance_from_wall/L_star
        y_hat = jnp.sqrt(x**2+y**2)-40/L_star #non-dim(distance) - radius/L_star
        D_omega_plus = jnp.maximum((2 / (sigma_omega2 * omega)) * (k_x * omega_x + k_y * omega_y), 10**-10)
        phi_1 = jnp.minimum(jnp.maximum(jnp.sqrt(k) / (0.09 * omega * y_hat), 500 / (Re * y_hat**2 * omega)),\
                    4 * k / (sigma_omega2 * D_omega_plus * y_hat**2))
        phi_2 = jnp.maximum( (2 * jnp.sqrt(k)) / (0.09 * omega * y_hat), 500 / (Re * y_hat**2 * omega))
        F1 = jnp.tanh(phi_1**4)
        F2 = jnp.tanh(phi_2)
        beta_i = F1 * beta_i1 + (1 - F1) * beta_i2
        alpha_star_0 = beta_i / 3
        alpha_infinity_1 = beta_i1 / beta_star_infinity - kappa**2\
            / (sigma_omega1 * jnp.sqrt(beta_star_infinity))
        alpha_infinity_2 = beta_i2 / beta_star_infinity - kappa**2\
            / (sigma_omega2 * jnp.sqrt(beta_star_infinity))
        alpha_infinity = F1 * alpha_infinity_1 + (1 - F1) * alpha_infinity_2
        Re_t = k / (mu * omega)
        alpha_star = alpha_star_infinity * (alpha_star_0 + Re_t / R_k) / (1 + Re_t / R_k)
        alpha = (alpha_infinity / alpha_star) * ((alpha_0 + Re_t / R_omega) / (1 + Re_t / R_omega))
        beta_star_i = beta_star_infinity * ((4/15 + (Re_t / R_beta)**4) / (1 + (Re_t / R_beta)**4))
        M_t = U_star*jnp.sqrt(2 * k / (gamma * R * T))
        F_Mt = jnp.where(M_t <= M_t0, 0, M_t**2 - M_t0**2)
        beta_star = beta_star_i * (1 + xi_star * F_Mt)
        beta = beta_i * (1 - beta_star_i / beta_i * xi_star * F_Mt)
        sigma_k = 1 / (F1 / sigma_k1 + (1 - F1) / sigma_k2)
        sigma_omega = 1 / (F1 / sigma_omega1 + (1 - F1) / sigma_omega2)
        S = jnp.sqrt(2 * ((u_x)**2 + (v_y)**2 + (1/2) * (u_y + v_x)**2))

        mu_t = k / omega * (1 / jnp.maximum(1 / alpha_star, S * F2 / (a1 * omega)))
        G_k = mu_t * S**2
        Y_k = beta_star * k * omega
        G_k_tilde = jnp.minimum(G_k, 10 * beta_star * k * omega)
        G_omega = alpha / mu_t * G_k_tilde
        Y_omega = beta * omega ** 2
        D_omega = 2 * (1 - F1) * (sigma_omega2 / omega) * (k_x * omega_x + k_y * omega_y)
        continuity = u_x + v_y

        def x_mom_grad1(params, t, x, y):
          return (1/Re+mu_t)*(4/3*u_x-2/3*v_y)
        def x_mom_grad2(params, t, x, y):
          return (1/Re+mu_t)*(u_y+v_x)
        x_momentum = u_t + 2*u*u_x + u*v_y + v*u_y + p_x \
          - grad(x_mom_grad1, argnums=2)(params, t, x, y) \
          - grad(x_mom_grad2, argnums=3)(params, t, x, y)
        def y_mom_grad1(params, t, x, y):
          return (1/Re+mu_t)*(4/3*v_y-2/3*u_x)
        def y_mom_grad2(params, t, x, y):
          return (1/Re+mu_t)*(u_y+v_x)
        y_momentum = v_t + v*u_x + u*v_x + 2*v*v_y + p_y \
          - grad(y_mom_grad1, argnums=3)(params, t, x, y) \
          - grad(y_mom_grad2, argnums=2)(params, t, x, y)
        def k_transport_grad1(params, t, x, y):
          return (1/Re + mu_t/sigma_k) * k_x
        def k_transport_grad2(params, t, x, y):
          return (1/Re + mu_t/sigma_k) * k_y
        k_transport = k_t + u*k_x + v*k_y \
          - grad(k_transport_grad1, argnums=2)(params, t, x, y) \
          - grad(k_transport_grad2, argnums=3)(params, t, x, y) \
          - G_k + Y_k
        def omega_transport_grad1(params, t, x, y):
          return (1/Re + mu_t/sigma_omega) * omega_x
        def omega_transport_grad2(params, t, x, y):
          return (1/Re + mu_t/sigma_omega) * omega_y
        omega_transport = omega_t + u*omega_x + v*omega_y \
          - grad(omega_transport_grad1, argnums=2)(params, t, x, y) \
          - grad(omega_transport_grad2, argnums=3)(params, t, x, y) \
          - G_omega + Y_omega - D_omega

        # outflow boundary residual
        u_out = u_x / self.Re - p
        v_out = v_x

        # symmetry boundary residual
        u_out = u_x / self.Re - p
        v_out = v_x
        # wall function for k and omega on walls is available (add feature)

        # symmetry boundary residual
        u_symmetry = u_y
        v_symmetry = v_y
        #k_symmetry = k_y
        #omega_symmetry = omega_y

        return continuity, x_momentum, y_momentum, k_transport, omega_transport,\
          u_out, v_out, u_symmetry, v_symmetry

    def continuity_net(self, params, t, x, y):
        print('continuity_net')
        continuity, _, _, _, _, _, _, _, _ = self.r_net(params, t, x, y)
        return continuity

    def x_momentum_net(self, params, t, x, y):
        print('x_momentum_net')
        _, x_momentum, _, _, _, _, _, _, _ = self.r_net(params, t, x, y)
        return x_momentum

    def y_momentum_net(self, params, t, x, y):
        print('y_momentum_net')
        _, _, y_momentum, _, _, _, _, _ ,_ = self.r_net(params, t, x, y)
        return y_momentum

    def k_transport_net(self, params, t, x, y):
        print('k_transport_net')
        _, _, _, k_transport, _, _, _, _ ,_ = self.r_net(params, t, x, y)
        return k_transport

    def omega_transport_net(self, params, t, x, y):
        print('omega_transport_net')
        _, _, _, _, omega_transport,_ , _ ,_ ,_ = self.r_net(params, t, x, y)
        return omega_transport

    def u_out_net(self, params, t, x, y):
        print('-u_out_net')
        _, _, _, _, _,u_out,_ ,_ ,_ = self.r_net(params, t, x, y)
        return u_out

    def v_out_net(self, params, t, x, y):
        print('-v_out_net')
        _, _, _, _, _, _,v_out,_ ,_ = self.r_net(params, t, x, y)
        return v_out

    def u_symmetry_net(self, params, t, x, y):
        print('u_symmetry_net')
        _, _, _, _, _, _, _, u_symmetry, _ = self.r_net(params, t, x, y)
        return u_symmetry

    def v_symmetry_net(self, params, t, x, y):
        print('v_symmetry_net')
        _, _, _, _, _, _, _, _, v_symmetry = self.r_net(params, t, x, y)
        return v_symmetry

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        continuity_pred, x_momentum_pred, y_momentum_pred, k_transport_pred,\
        omega_transport_pred, _, _, _, _ = self.r_pred_fn(\
            params, t_sorted, batch[:, 1], batch[:, 2])

        continuity_pred = continuity_pred.reshape(self.num_chunks, -1)
        x_momentum_pred = x_momentum_pred.reshape(self.num_chunks, -1)
        y_momentum_pred = y_momentum_pred.reshape(self.num_chunks, -1)
        k_transport_pred = k_transport_pred.reshape(self.num_chunks, -1)
        omega_transport_pred = omega_transport_pred.reshape(self.num_chunks, -1)

        continuity_l = jnp.mean(continuity_pred**2, axis=1)
        x_momentum_l = jnp.mean(x_momentum_pred**2, axis=1)
        y_momentum_l = jnp.mean(y_momentum_pred**2, axis=1)
        k_transport_l = jnp.mean(k_transport_pred**2, axis=1)
        omega_transport_l = jnp.mean(omega_transport_pred**2, axis=1)

        continuity_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ continuity_l)))
        x_momentum_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ x_momentum_l)))
        y_momentum_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ y_momentum_l)))
        k_transport_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ k_transport_l)))
        omega_transport_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ omega_transport_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([continuity_gamma, x_momentum_gamma,
        y_momentum_gamma, k_transport_gamma, omega_transport_gamma])
        gamma = gamma.min(0)

        return continuity_l, x_momentum_l, y_momentum_l, k_transport_l,\
        omega_transport_l, gamma

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        # Unpack batch
        ic_batch = batch["ic"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        symmetry_batch = batch["symmetry"] #me
        res_batch = batch["res"]

        coords_batch, u_batch, v_batch, p_batch, k_batch, omega_batch = ic_batch

        u_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.u_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        v_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.v_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        p_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.p_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        k_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.k_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        w_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.omega_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )

        u_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.u_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
        )

        u_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.u_out_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
        )
        v_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_out_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
        )

        u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.u_net,
            params,
            noslip_batch[:, 0],
            noslip_batch[:, 1],
            noslip_batch[:, 2],
        )
        v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_net,
            params,
            noslip_batch[:, 0],
            noslip_batch[:, 1],
            noslip_batch[:, 2],
        )

        u_symmetry_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.u_symmetry_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
        )
        v_symmetry_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_symmetry_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            res_batch = jnp.array(
                [res_batch[:, 0].sort(), res_batch[:, 1], res_batch[:, 2]]
            ).T
            continuity_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.continuity_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            x_momentum_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.x_momentum_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            y_momentum_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.y_momentum_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            k_transport_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.k_momentum_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            omega_transport_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.omega_momentum_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )

            continuity_ntk = continuity_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            x_momentum_ntk = x_momentum_ntk.reshape(self.num_chunks, -1)
            y_momentum_ntk = y_momentum_ntk.reshape(self.num_chunks, -1)
            k_transport_ntk = k_transport_ntk.reshape(self.num_chunks, -1)
            omega_transport_ntk = omega_transport_ntk.reshape(self.num_chunks, -1)

            # average convergence rate over each chunk
            continuity_ntk = jnp.mean(continuity_ntk, axis=1)
            x_momentum_ntk = jnp.mean(x_momentum_ntk, axis=1)
            y_momentum_ntk = jnp.mean(y_momentum_ntk, axis=1)
            k_transport_ntk = jnp.mean(k_transport_ntk, axis=1)
            omega_transport_ntk = jnp.mean(omega_transport_ntk, axis=1)

            _, _, _, _, _, causal_weights = self.res_and_w(params, res_batch)
            continuity_ntk = continuity_ntk * causal_weights  # multiply by causal weights
            x_momentum_ntk = x_momentum_ntk * causal_weights
            y_momentum_ntk = y_momentum_ntk * causal_weights
            k_transport_ntk = k_transport_ntk * causal_weights
            omega_transport_ntk = omega_transport_ntk * causal_weights
        else:
            continuity_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.continuity_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            x_momentum_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.x_momentum_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            y_momentum_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.y_momentum_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            k_transport_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.k_transport_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            omega_transport_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.omega_transport_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )

        ntk_dict = {
            "u_ic": u_ic_ntk,
            "v_ic": v_ic_ntk,
            "p_ic": p_ic_ntk,
            "k_ic": k_ic_ntk,
            "omega_ic": omega_ic_ntk,
            "u_in": u_in_ntk,
            "v_in": v_in_ntk,
            "u_out": u_out_ntk,
            "v_out": v_out_ntk,
            "u_noslip": u_noslip_ntk,
            "v_noslip": v_noslip_ntk,
            "u_symmetry": u_symmetry_ntk,
            "v_symmetry": v_symmetry_ntk,
            "continuity": continuity_ntk,
            "x_momentum": x_momentum_ntk,
            "y_momentum": y_momentum_ntk,
            "k_transport": k_transport_ntk,
            "omega_transport": omega_transport_ntk,
        }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Unpack batch
        ic_batch = batch["ic"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        symmetry_batch = batch["symmetry"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, u_batch, v_batch, p_batch, k_batch, omega_batch = ic_batch

        u_ic_pred = self.u0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        v_ic_pred = self.v0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        p_ic_pred = self.p0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        k_ic_pred = self.k0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        omega_ic_pred = self.omega0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])

        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        p_ic_loss = jnp.mean((p_ic_pred - p_batch) ** 2)
        k_ic_loss = jnp.mean((k_ic_pred - p_batch) ** 2)
        omega_ic_loss = jnp.mean((omega_ic_pred - omega_batch) ** 2)

        # inflow loss
        u_in, _ = self.inflow_fn(inflow_batch[:, 2])

        u_in_pred = self.u_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2]
        )
        v_in_pred = self.v_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2]
        )

        u_in_loss = jnp.mean((u_in_pred - u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred**2)

        # outflow loss
        _, _, _, _, _, u_out_pred, v_out_pred, _, _ = self.r_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], outflow_batch[:, 2]
        )

        u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)

        # noslip loss
        u_noslip_pred = self.u_pred_fn(
            params, noslip_batch[:, 0], noslip_batch[:, 1], noslip_batch[:, 2]
        )
        v_noslip_pred = self.v_pred_fn(
            params, noslip_batch[:, 0], noslip_batch[:, 1], noslip_batch[:, 2]
        )

        u_noslip_loss = jnp.mean(u_noslip_pred**2)
        v_noslip_loss = jnp.mean(v_noslip_pred**2)

        # symmetry loss
        _, _, _, _, _, _, _, u_symmetry_pred, v_symmetry_pred = self.r_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], outflow_batch[:, 2]
        )

        u_symmetry_loss = jnp.mean(u_symmetry_pred**2)
        v_symmetry_loss = jnp.mean(v_symmetry_pred**2)

        # residual loss
        if self.config.weighting.use_causal == True:
            continuity_l, x_momentum_l, y_momentum_l, k_transport_l,\
            omega_transport_l, gamma = self.res_and_w(params, res_batch)
            continuity_loss = jnp.mean(gamma * continuity_l)
            x_momentum_loss = jnp.mean(gamma * x_momentum_l)
            y_momentum_loss = jnp.mean(gamma * y_momentum_l)
            k_transport_loss = jnp.mean(gamma * k_transport_l)
            omega_transport_loss = jnp.mean(gamma * omega_transport_l)

        else:
            continuity_pred, x_momentum_pred, y_momentum_pred,
            k_transport_pred, omega_transport_pred, _, _, _, _ = self.r_pred_fn(\
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2])
            continuity_loss = jnp.mean(continuity_pred**2)
            x_momentum_loss = jnp.mean(x_momentum_pred**2)
            y_momentum_loss = jnp.mean(y_momentum_pred**2)
            k_transport_loss = jnp.mean(k_transport_pred**2)
            omega_transport_loss = jnp.mean(omega_transport_pred**2)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "p_ic": p_ic_loss,
            "k_ic": k_ic_loss,
            "omega_ic": omega_ic_loss,
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "u_out": u_out_loss,
            "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
            "u_symmetry": u_symmetry_loss,
            "v_symmetry": v_symmetry_loss,
            "continuity": continuity_loss,
            "x_momentum": x_momentum_loss,
            "y_momentum": y_momentum_loss,
            "k_transport": k_transport_loss,
            "omega_transport": omega_transport_loss,
        }

        return loss_dict

    def u_v_grads(self, params, t, x, y):
        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)

        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)

        return u_x, v_x, u_y, v_y

    @partial(jit, static_argnums=(0,))
    def compute_drag_lift(self, params, t, U_star, L_star):
        nu = 0.000015  # Dimensional viscosity
        radius = 80.0  # radius of cylinder
        center = (0.0, 0.0)  # center of cylinder
        num_theta = 256  # number of points on cylinder for evaluation

        # Discretize cylinder into points
        theta = jnp.linspace(0.0, 2 * jnp.pi, num_theta)
        d_theta = theta[1] - theta[0]
        ds = radius * d_theta

        # Cylinder coordinates
        x_cyl = radius * jnp.cos(theta) + center[0]
        y_cyl = radius * jnp.sin(theta) + center[1]

        # Out normals of cylinder
        n_x = jnp.cos(theta)
        n_y = jnp.sin(theta)

        # Nondimensionalize input cylinder coordinates
        x_cyl = x_cyl / L_star
        y_cyl = y_cyl / L_star

        # Nondimensionalize front and back points
        front = jnp.array([center[0] - radius, center[1]]) / L_star
        back = jnp.array([center[0] + radius, center[1]]) / L_star

        # Predictions
        u_x_pred, v_x_pred, u_y_pred, v_y_pred = vmap(
            vmap(self.u_v_grads, (None, None, 0, 0)), (None, 0, None, None)
        )(params, t, x_cyl, y_cyl)

        p_pred = vmap(vmap(self.p_net, (None, None, 0, 0)), (None, 0, None, None))(
            params, t, x_cyl, y_cyl
        )

        p_pred = p_pred - jnp.mean(p_pred, axis=1, keepdims=True)

        p_front_pred = vmap(self.p_net, (None, 0, None, None))(
            params, t, front[0], front[1]
        )
        p_back_pred = vmap(self.p_net, (None, 0, None, None))(
            params, t, back[0], back[1]
        )
        p_diff = p_front_pred - p_back_pred

        # Dimensionalize velocity gradients and pressure
        u_x_pred = u_x_pred * U_star / L_star
        v_x_pred = v_x_pred * U_star / L_star
        u_y_pred = u_y_pred * U_star / L_star
        v_y_pred = v_y_pred * U_star / L_star
        p_pred = p_pred * U_star**2
        p_diff = p_diff * U_star**2

        I0 = (-p_pred[:, :-1] + 2 * nu * u_x_pred[:, :-1]) * n_x[:-1] + nu * (
            u_y_pred[:, :-1] + v_x_pred[:, :-1]
        ) * n_y[:-1]
        I1 = (-p_pred[:, 1:] + 2 * nu * u_x_pred[:, 1:]) * n_x[1:] + nu * (
            u_y_pred[:, 1:] + v_x_pred[:, 1:]
        ) * n_y[1:]

        F_D = 0.5 * jnp.sum(I0 + I1, axis=1) * ds

        I0 = (-p_pred[:, :-1] + 2 * nu * v_y_pred[:, :-1]) * n_y[:-1] + nu * (
            u_y_pred[:, :-1] + v_x_pred[:, :-1]
        ) * n_x[:-1]
        I1 = (-p_pred[:, 1:] + 2 * nu * v_y_pred[:, 1:]) * n_y[1:] + nu * (
            u_y_pred[:, 1:] + v_x_pred[:, 1:]
        ) * n_x[1:]

        F_L = 0.5 * jnp.sum(I0 + I1, axis=1) * ds

        # Nondimensionalized drag and lift and pressure difference
        C_D = 2 / (U_star**2 * L_star) * F_D
        C_L = 2 / (U_star**2 * L_star) * F_L

        return C_D, C_L, p_diff


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    # def log_preds(self, params, x_star, y_star):
    #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
    #
    #     fig = plt.figure()
    #     plt.pcolor(U_pred.T, cmap='jet')
    #     log_dict['U_pred'] = fig
    #     fig.close()

    def __call__(self, state, batch):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, _, _, _, causal_weight = self.model.res_and_w(state.params, batch["res"])
            self.log_dict["cas_weight"] = causal_weight.min()

        # if self.config.logging.log_errors:
        #     self.log_errors(state.params, coords, u_ref, v_ref)
        #
        # if self.config.logging.log_preds:
        #     self.log_preds(state.params, coords)

        return self.log_dict
