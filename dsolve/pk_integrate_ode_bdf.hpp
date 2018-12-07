#ifndef STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_BDF_HPP
#define STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_BDF_HPP

#include <stan/math/torsten/dsolve/pk_cvodes_integrator.hpp>
#include <boost/mpi.hpp>
#include <stan/math/torsten/mpi.hpp>
#include <ostream>
#include <vector>

namespace torsten {
namespace dsolve {

  /*
   * integrate ODE using Torsten's BDF implementation
   * based on CVODES.
   * @tparam F ODE RHS functor type
   * @tparam Tt time type
   * @tparam T_initial initial condition type
   * @tparam T_param parameters(theta) type
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
                         long int max_num_step = 1e6) {  // NOLINT(runtime/int)
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

#ifdef TORSTEN_MPI
  /* MPI version */
  template <typename F, typename Tt, typename T_initial, typename T_param>
  std::vector<std::vector<std::vector<typename stan::return_type<Tt,
                                                                 T_initial,
                                                                 T_param>::type> > > // NOLINT
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
                       long int max_num_step = 1e6) {  // NOLINT(runtime/int)
    using torsten::dsolve::PKCvodesFwdSystem;
    using torsten::dsolve::PKCvodesIntegrator;
    using torsten::PkCvodesSensMethod;
    using Ode = PKCvodesFwdSystem<F, Tt, T_initial, T_param, CV_BDF, AD>;
    const int m = theta[0].size();
    const int n = y0[0].size();
    const int np = theta.size(); // population size

    PKCvodesService<typename Ode::Ode> serv(n, m);
    PKCvodesIntegrator solver(rtol, atol, max_num_step);
    
    // make sure MPI is on
    boost::mpi::environment env;
    boost::mpi::communicator world;

    for (int i = 0; i < np; ++i) {
      if(torsten::mpi::is_mine(world, i, np)) {
        Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
        auto res = solver.integrate(ode);
      }
    }

}
#endif
}
}
#endif
