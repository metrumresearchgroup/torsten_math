#ifndef STAN_MATH_TORSTEN_ODE_CKRK_HPP
#define STAN_MATH_TORSTEN_ODE_CKRK_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_integrator.hpp>
#include <stan/math/torsten/dsolve/pmx_odeint_integrator.hpp>
#include <ostream>
#include <vector>

namespace torsten {
  /**
   * solve an ODE given its RHS with Boost Odeint's Ckrk solver.
   *
   * @tparam F Functor type for RHS of ODE
   * @tparam Tt type of time
   * @tparam T_y0 type of initial condition @c y0
   * @tparam T_param type of parameter @c theta
   * @param f RHS functor of ODE system
   * @param y0 initial condition
   * @param t0 initial time
   * @param ts time steps
   * @param theta parameters for ODE
   * @param x_r data used in ODE
   * @param x_i integer data used in ODE
   * @param msgs output stream
   * @param rtol relative tolerance
   * @param atol absolute tolerance
   * @param max_num_step maximum number of integration steps allowed.
   * @return a vector of vectors for results in each time step.
   */
  template <typename F, typename Tt, typename T_y0, typename... T_param,
	    stan::require_eigen_col_vector_t<T_y0>* = nullptr>
  std::vector<Eigen::Matrix<typename stan::return_type_t<Tt, T_y0, T_param...>,-1,1> >
  pmx_ode_ckrk_ctrl(const F& f,
                    const T_y0& y0,
                    double t0,
                    const std::vector<Tt>& ts,
                    double rtol, double atol, int max_num_step,
                    std::ostream* msgs,
                    const T_param&... args) {
    using dsolve::PMXOdeIntegrator;
    using dsolve::PMXOdeintIntegrator;
    PMXOdeIntegrator<dsolve::PMXVariadicOdeSystem, PMXOdeintIntegrator<dsolve::odeint_scheme_ckrk>> solver(rtol, atol, max_num_step, msgs);
    Eigen::Matrix<stan::return_type_t<T_y0>, -1, 1> y0_vec(y0);
    return solver(f, y0_vec, t0, ts, args...);
  }

  /**
   * overload with default ode controls
   */
  template <typename F, typename Tt, typename T_y0, typename... T_param>
  std::vector<Eigen::Matrix<typename stan::return_type_t<Tt, T_y0, T_param...>,-1,1> >
  pmx_ode_ckrk(const F& f,
               const T_y0& y0,
               double t0,
               const std::vector<Tt>& ts,
               std::ostream* msgs,
               const T_param&... args) {
    return pmx_ode_ckrk_ctrl(f, y0, t0, ts, 1.e-6, 1.e-6, 1e6, msgs, args...);
  }
}
#endif
