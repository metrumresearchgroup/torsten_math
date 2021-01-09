#ifndef STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_ERK45_HPP
#define STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_ERK45_HPP

#include <stan/math/torsten/dsolve/pmx_ode_integrator.hpp>
#include <stan/math/torsten/dsolve/arkode_service.hpp>
#include <stan/math/torsten/dsolve/pmx_arkode_system.hpp>
#include <stan/math/torsten/dsolve/pmx_arkode_integrator.hpp>
#include <stan/math/torsten/dsolve/ode_check.hpp>
#include <arkode/arkode_erkstep.h>
#include <ostream>
#include <vector>

namespace torsten {
  /*
   * solve an ODE given its RHS with Boost Odeint's Erk45 solver.
   *
   * @tparam F Functor type for RHS of ODE
   * @tparam Tt type of time
   * @tparam T_initial type of initial condition @c y0
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
  template <typename F, typename Tt, typename T_initial, typename T_param>
  inline std::vector<std::vector<typename stan::return_type<Tt, T_initial, T_param>::type> >
  pmx_integrate_ode_erk45(const F& f,
                         const std::vector<T_initial>& y0,
                         double t0,
                         const std::vector<Tt>& ts,
                         const std::vector<T_param>& theta,
                         const std::vector<double>& x_r,
                         const std::vector<int>& x_i,
                         double rtol,
                         double atol,
                         long int max_num_step,
                         std::ostream* msgs = nullptr) {
    using dsolve::PMXOdeIntegrator;
    using dsolve::PMXArkodeIntegrator;
    using dsolve::PMXArkodeSystem;
    PMXOdeIntegrator<PMXArkodeSystem,
                     PMXArkodeIntegrator<DORMAND_PRINCE_7_4_5>> solver(rtol, atol, max_num_step, msgs);
    return solver(f, y0, t0, ts, theta, x_r, x_i);
  }

  /*
   * overload with default ode controls
   */
  template <typename F, typename Tt, typename T_initial, typename T_param>
  inline std::vector<std::vector<typename stan::return_type<Tt, T_initial, T_param>::type> >
  pmx_integrate_ode_erk45(const F& f,
                         const std::vector<T_initial>& y0,
                         double t0,
                         const std::vector<Tt>& ts,
                         const std::vector<T_param>& theta,
                         const std::vector<double>& x_r,
                         const std::vector<int>& x_i,
                         std::ostream* msgs = nullptr) {
    return pmx_integrate_ode_erk45(f, y0, t0, ts, theta, x_r, x_i,
                                  1.e-6, 1.e-6, 1e5,
                                  msgs);
  }
}
#endif
