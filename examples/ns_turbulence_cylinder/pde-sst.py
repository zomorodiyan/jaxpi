# Constants (SST k-omega)
a_1 = 0.31
beta_i1 = 0.075
alpha_0 = 1/9
beta_i2 = 0.0828
beta_star_inf = 0.09
sigma_omega_1 = 2.0
R_beta = 8
sigma_omega_2 = 1.168
R_k = 6
sigma_k1 = 1.176
R_omega = 2.95
sigma_k2 = 1.0
xi_star = 1.5
M_t0 = 0.25
f_beta = 1.0
f_star_beta = 1
kappa = 0.41

# Physical Constants
rho = 1.225  # kg/m^3
mu = 1.81e-5  # kg/(m.s)
gamma = 1.4
R = 287  # m^2/(s^2.K)
T = 297.15  # K
L_star = 80 #m
U_star = 9.0 #m/s
Re = rho*U_star*L_star/mu

def r_net(self, params, t, x, y):
  u, v, p, k, omega = self.neural_net(params, t, x, y)
  # Compute the partial derivatives using JAX
  u_t = grad(self.u_net, argnums=1)(params, t, x, y)
  u_x = grad(self.u_net, argnums=2)(params, t, x, y)
  u_y = grad(self.u_net, argnums=3)(params, t, x, y)
  u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
  u_xy = grad(grad(self.u_net, argnums=2), argnums=3)(params, t, x, y)
  v_t = grad(self.v_net, argnums=1)(params, t, x, y)
  v_x = grad(self.v_net, argnums=2)(params, t, x, y)
  v_y = grad(self.v_net, argnums=3)(params, t, x, y)
  v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y)
  v_yx = grad(grad(self.v_net, argnums=3), argnums=2)(params, t, x, y)
  p_x = grad(self.v_net, argnums=2)(params, t, x, y)
  p_y = grad(self.v_net, argnums=3)(params, t, x, y)
  k_t = grad(k_net, argnums=1)(params, t, x, y_coord)
  k_x = grad(k_net, argnums=2)(params, t, x, y_coord)
  k_y = grad(k_net, argnums=3)(params, t, x, y_coord)
  omega_t = grad(omega_net, argnums=1)(params, t, x, y_coord)
  omega_x = grad(omega_net, argnums=2)(params, t, x, y_coord)
  omega_y = grad(omega_net, argnums=3)(params, t, x, y_coord)
  #intermediates
  S = (2*( u_x**2+ v_y**2+ 0.5(u_y+v_x)**2))**0.5
  D_omega_plus = max(2/(sigma_omega_2*omega)*(k_x*omega_x + k_y*omega_y), 1e-10)
  phi_1 = min(max(k**0.5/(0.09*omega*y_hat), 500/(Re*y_hat*omega)),
  4*k/(sigma_omega_2 * D_omega_plus * y_hat**2)
  phi_2 = max(2*k**0.5/(0.09*omega*y_hat), 500/(Re*y_hat*omega))
  F_1 = tanh(phi_1**4)
  F_2 = tanh(phi_2)
  alpha_inf1 = beta_i1 / beta_star_inf - kappa ** 2 / (sigma_omega_1 * beta_star_inf ** 0.5)
  alpha_inf2 = beta_i2 / beta_star_inf - kappa ** 2 / (sigma_omega_2 * beta_star_inf ** 0.5)
  alpha_inf = F_1 * alpha_inf1 + (1 - F_1) * alpha_inf2
  Re_t = Re * k / omega
  alpha_star = alpha_star_inf * (alpha_1 + Re_t / R_k) / (1 + Re_t / R_k)
  mu_t = k / omega / max(1 / alpha_star, S * F_2 / (a_1 * omega))
  Mt = U_star*(2*k/(gamma*R*T))**0.5
  beta_star_i = beta_star_inf*((4/15 + (Re_t/R_beta)**4)/(1+(Re_t/R_beta)**4))
  F_Mt = lambda M_t, M_t0: 0 if M_t <= M_t0 else M_t**2 - M_t0**2
  beta_star = beta_star_i(1 + xi_star*F_Mt)
  G_k = mu_t*S**2
  G_tilde_k = min(G_k, 10*beta_star*k*omega)
  Y_k = beta_star*k*omega
  alpha_inf = F_1*alpha_inf_1 + (1-F_1)*alpha_inf_2
  alpha = alpha_inf/alpha_star*(alpha_0+Re_t/R_omega)/(1+Re_t/R_omega)
  beta_i = F_1*beta_i1 + (1-F_1)*beta_i2
  beta = beta_i*(1-beta_star_i/beta_i*xi_star*F_Mt)
  G_omega = alpha/mu_t*G_tilde_k
  Y_omega = beta/omega**2
  D_omega = 2*(1-F_1)*sigma_omega_2/omega*(k_x*omega_x+k_y*omega_y)
  sigma_k = 1/(F_1/sigma_k_1 +(1-F_1)/sigma_k_2)
  sigma_omega = 1/(F_1/sigma_omega_1 +(1-F_1)/sigma_omega_2)
  #PDEs
  continutiy = u_x + v_y
  x_momentum = u_t + 2*u*u_x + u*v_y + v*u_y + p_x
    - grad((1/Re+mu_t)*(2*u_x-2/3*(u_x+v_y)), argnums=2)(params, t, x, y)
    - grad((1/Re+mu_t)*(u_y+v_x), argnums=3)(params, t, x, y)
  y_momentum = u_t + 2*v*v_y + v*u_x + u*v_x + p_y
    - grad((1/Re+mu_t)*(2*v_y-2/3*(u_x+v_y)), argnums=3)(params, t, x, y)
    - grad((1/Re+mu_t)*(u_y+v_x), argnums=2)(params, t, x, y)
  k-transport = k_t
    + grad(u*k, argnums=2)(params, t, x, y)
    + grad(u*k, argnums=3)(params, t, x, y)
    - grad((1/Re+mu_t/sigma_k)*k_x, argnums=2)(params, t, x, y)
    - grad((1/Re+mu_t/sigma_k)*k_y, argnums=3)(params, t, x, y)
    - G_tilde_k + Y_k
  omega-transport = omega_t
    + grad(u*omega, argnums=2)(params, t, x, y)
    + grad(v*omega, argnums=3)(params, t, x, y)
    - grad((1/Re+mu_t/sigma_k)*k_x, argnums=2)(params, t, x, y)
    - grad((1/Re+mu_t/sigma_k)*k_y, argnums=3)(params, t, x, y)
    - G_omega + Y_omega - D_omega
