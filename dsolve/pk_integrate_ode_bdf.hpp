#ifndef STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_BDF_HPP
#define STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_BDF_HPP

#include <stan/math/prim/scal/err/check_greater.hpp>
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
  /**
   * Solve population ODE model by delegating the population
   * ODE integration task to multiple processors through
   * MPI. When all input are data, we simply collect them.
   * 
   * @return res nested vector that contains results for
   * (individual i, time j, equation k)
   **/
  template<typename F>
  std::vector<std::vector<std::vector<double> > >
  pk_integrate_ode_bdf(const F& f,
                       const std::vector<std::vector<double> >& y0,
                       double t0,
                       const std::vector<std::vector<double> >& ts,
                       const std::vector<std::vector<double> >& theta,
                       const std::vector<std::vector<double> >& x_r,
                       const std::vector<std::vector<int> >& x_i,
                       std::ostream* msgs = nullptr,
                       double rtol = 1e-10,
                       double atol = 1e-10,
                       long int max_num_step = 1e6) {  // NOLINT(runtime/int)
    using stan::math::var;
    using std::vector;
    using torsten::dsolve::PKCvodesFwdSystem;
    using torsten::dsolve::PKCvodesIntegrator;
    using torsten::PkCvodesSensMethod;
    using Ode = PKCvodesFwdSystem<F, double, double, double, CV_BDF, AD>;
    const int m = theta[0].size();
    const int n = y0[0].size();
    const int np = theta.size(); // population size

    PKCvodesService<typename Ode::Ode> serv(n, m);
    PKCvodesIntegrator solver(rtol, atol, max_num_step);
    
    // make sure MPI is on
    int intialized;
    MPI_Initialized(&intialized);
    stan::math::check_greater("pk_integrate_ode_bdf", "MPI_Intialized", intialized, 0);

    boost::mpi::communicator world;

    Eigen::MatrixXd res_i;
    vector<vector<vector<double>> > res(np);
    int ns, nsol, nsys, nt;

    for (int i = 0; i < np; ++i) {
      int my_worker_id = torsten::mpi::my_worker(i, np, world.size());
      Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
      ns   = ode.ns();
      nsol = ode.n_sol();
      nsys = ode.n_sys();
      nt   = ode.ts().size();
      res_i.resize(nt, nsys);
      if(world.rank() == my_worker_id) {
        res_i = solver.integrate<Ode, false>(ode);
      }
      broadcast(world, res_i.data(), res_i.size(), my_worker_id);
      res[i].resize(nt);
      for (int j = 0 ; j < nt; ++j) {
        res[i][j].resize(nsys);
        for (int k = 0; k < nsys; ++k) {
          res[i][j][k] = res_i(j, k);
        }
      }
    }
    return res;
  }

  // template<typename T>
  // T pk_ode_assemble_solution(double & sol, std::vector<T>& vars, std::vector<double>& g) { // NOLINT(runtime/int)
  //   return precomputed_gradients(sol, vars, g);
  // }
                             
  // template<>
  // double pk_ode_assemble_solution(double & sol, std::vector<double>& vars, std::vector<double>& g) { // NOLINT(runtime/int)
  //   return sol;
  // }

  /**
   * Solve population ODE model by delegating the population
   * ODE integration task to multiple processors through
   * MPI, then gather the results, before generating @c var arrays.
   *
   * @return res nested vector that contains results for
   * (individual i, time j, equation k)
   **/
  template <typename F, typename Tt, typename T_initial, typename T_param>
  std::vector<std::vector<std::vector<typename stan::return_type<Tt,
                                                                 T_initial,
                                                                 T_param>::type> > >
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
    using std::vector;
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
    int intialized;
    MPI_Initialized(&intialized);
    stan::math::check_greater("pk_integrate_ode_bdf", "MPI_Intialized", intialized, 0);

    boost::mpi::communicator world;

    using scalar_type = typename stan::return_type<Tt, T_initial, T_param>::type;

    Eigen::MatrixXd res_i;
    vector<vector<vector<scalar_type>> > res(np);
    vector<scalar_type> vars;
    std::vector<double> g;
    int ns, nsol, nsys, nt;

    for (int i = 0; i < np; ++i) {
      int my_worker_id = torsten::mpi::my_worker(i, np, world.size());
      Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
      vars = ode.vars();
      ns   = ode.ns();
      nsys = ode.n_sys();
      nt   = ode.ts().size();
      nsol = ode.n_sol();
      res_i.resize(nt, nsys);
      if(world.rank() == my_worker_id) {
        res_i = solver.integrate<Ode, false>(ode);
      }
      broadcast(world, res_i.data(), res_i.size(), my_worker_id);
      g.resize(ns);
      res[i].resize(nt);
      for (int j = 0 ; j < nt; ++j) {
        res[i][j].resize(nsys);
        for (int k = 0; k < n; ++k) {
          for (int l = 0 ; l < ns; ++l) g[l] = res_i(j, k * nsol + l + 1);
          res[i][j][k] = precomputed_gradients(res_i(j, k * nsol), vars, g);
        }
      }
    }
    return res;
}
#endif
}
}
#endif
