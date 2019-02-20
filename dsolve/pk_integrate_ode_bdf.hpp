#ifndef STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_BDF_HPP
#define STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_BDF_HPP

#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_integrator.hpp>
#include <stan/math/torsten/mpi.hpp>
#include <stan/math/torsten/mpi/pk_population_integrator.hpp>

#include <mpi.h>
#include <ostream>
#include <vector>

namespace torsten {
namespace dsolve {

  /*
   * solve an ODE given its RHS with CVODES' BDF solver.
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
  pk_integrate_ode_bdf(const F& f,
                         const std::vector<T_initial>& y0,
                         double t0,
                         const std::vector<Tt>& ts,
                         const std::vector<T_param>& theta,
                         const std::vector<double>& x_r,
                         const std::vector<int>& x_i,
                         std::ostream* msgs = nullptr,
                         double rtol = 1e-10,
                         double atol = 1e-10,
                         long int max_num_step = 1e8) {
    using torsten::dsolve::PKCvodesFwdSystem;
    using torsten::dsolve::PKCvodesIntegrator;
    using torsten::PkCvodesSensMethod;
    using Ode = PKCvodesFwdSystem<F, Tt, T_initial, T_param, CV_BDF, AD>;
    const int m = theta.size();
    const int n = y0.size();

    PKCvodesService<typename Ode::Ode> serv(n, m);

    Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
    PKCvodesIntegrator solver(rtol, atol, max_num_step);
    return solver.integrate(ode);
}

  /**
   * Solve population ODE model by delegating the population
   * ODE integration task to multiple processors through
   * MPI, then gather the results, before generating @c var arrays.
   * Each entry has an additional level of nested vector to
   * identifiy the individual among a population of ODE parameters/data.
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
   * @return res nested vector that contains results for
   * (individual i, time j, equation k)
   **/
  template <typename F, typename Tt, typename T_initial, typename T_param>
  std::vector<Eigen::Matrix<typename stan::return_type<Tt, T_initial, T_param>::type, // NOLINT
                            Eigen::Dynamic, Eigen::Dynamic> >
  pk_integrate_ode_bdf(const F& f,
                       const std::vector<std::vector<T_initial> >& y0,
                       double t0,
                       const std::vector<std::vector<Tt> >& ts,
                       const std::vector<std::vector<T_param> >& theta,
                       const std::vector<std::vector<double> >& x_r,
                       const std::vector<std::vector<int> >& x_i,
                       std::ostream* msgs = nullptr,
                       double rtol = 1e-10,
                       double atol = 1e-10,
                       long int max_num_step = 1e8) {  // NOLINT(runtime/int)
    static const char* caller("pk_integrate_ode_bdf");
    stan::math::check_consistent_sizes(caller, "y0", y0, "ts",     ts);
    stan::math::check_consistent_sizes(caller, "y0", y0, "theta",  theta);
    stan::math::check_consistent_sizes(caller, "y0", y0, "x_r",    x_r);
    stan::math::check_consistent_sizes(caller, "y0", y0, "x_i",    x_i);

    PKCvodesIntegrator solver(rtol, atol, max_num_step);

    torsten::mpi::PkPopulationIntegrator<F, CV_BDF> mpi_solver(solver);

    return mpi_solver(f, y0, t0, ts, theta, x_r, x_i, msgs);
}

}
}
#endif
