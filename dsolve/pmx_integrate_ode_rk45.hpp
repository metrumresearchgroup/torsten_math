#ifndef STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_rk45_HPP
#define STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_rk45_HPP

#include <stan/math/torsten/dsolve/pmx_odeint_system.hpp>
#include <stan/math/torsten/mpi.hpp>
#include <stan/math/torsten/dsolve/ode_check.hpp>
#include <ostream>
#include <vector>

namespace torsten {
  /*
   * solve an ODE given its RHS with Boost Odeint's Rk45 solver.
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
  std::vector<std::vector<typename stan::return_type<Tt,
                                                     T_initial,
                                                     T_param>::type> >
  pmx_integrate_ode_rk45(const F& f,
                         const std::vector<T_initial>& y0,
                         double t0,
                         const std::vector<Tt>& ts,
                         const std::vector<T_param>& theta,
                         const std::vector<double>& x_r,
                         const std::vector<int>& x_i,
                         std::ostream* msgs = nullptr,
                         double rtol = 1e-6,
                         double atol = 1e-6,
                         long int max_num_step = 1e6) {
    static const char* caller = "pmx_integrate_ode_rk45";
    dsolve::ode_check(y0, t0, ts, theta, x_r, x_i, caller);

    using Ode = dsolve::PMXOdeintSystem<F, T_initial, T_param>;
    using solver_t = boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>, double, std::vector<double>, double>;
    const int n = y0.size();
    const int m = theta.size();

    // static dsolve::PMXCvodesService<typename Ode::Ode> serv(n, m);
    Ode ode{f, t0, ts, y0, theta, x_r, x_i, msgs};

    std::vector<double> ts_vec(ts.size() + 1);
    ts_vec[0] = t0;
    std::copy(ts.begin(), ts.end(), ts_vec.begin() + 1);

    const double init_dt = 0.1;
    integrate_times(make_dense_output(atol, rtol, solver_t()),
                    boost::ref(ode), ode.y0_fwd_system,
                    ts_vec.begin(), ts_vec.end(),
                    init_dt, boost::ref(ode),
                    boost::numeric::odeint::max_step_checker(max_num_step));

    // remove the first state corresponding to the initial value
    // y_coupled.erase(y_coupled.begin());

    // the coupled system also encapsulates the decoupling operation
    return ode.decouple_states(ode.y_res_);

    // dsolve::PMXCvodesIntegrator solver(rtol, atol, max_num_step);
    // return solver.integrate(ode);
}
}
#endif
