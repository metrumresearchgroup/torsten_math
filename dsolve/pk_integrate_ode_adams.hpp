#ifndef STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_ADAMS_HPP
#define STAN_MATH_TORSTEN_DSOLVE_INTEGRATE_ODE_ADAMS_HPP

#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_integrator.hpp>
#include <mpi.h>
#include <stan/math/torsten/mpi.hpp>
#include <ostream>
#include <vector>

namespace torsten {
namespace dsolve {

  template <typename F, typename Tt, typename T_initial, typename T_param>
  std::vector<std::vector<typename stan::return_type<Tt,
                                                     T_initial,
                                                     T_param>::type> >
  pk_integrate_ode_adams(const F& f,
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
    using Ode = PKCvodesFwdSystem<F, Tt, T_initial, T_param, CV_ADAMS, AD>;
    const int n = y0.size();
    const int m = theta.size();

    static PKCvodesService<typename Ode::Ode> serv(n, m);

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
  std::vector<Eigen::MatrixXd>
  // std::vector<std::vector<std::vector<double> > >
  pk_integrate_ode_adams(const F& f,
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
    using Ode = PKCvodesFwdSystem<F, double, double, double, CV_ADAMS, AD>;
    const int m = theta[0].size();
    const int n = y0[0].size();
    const int np = theta.size(); // population size

    PKCvodesService<typename Ode::Ode> serv(n, m);
    PKCvodesIntegrator solver(rtol, atol, max_num_step);
    
    // make sure MPI is on
    int intialized;
    MPI_Initialized(&intialized);
    stan::math::check_greater("pk_integrate_ode_bdf", "MPI_Intialized", intialized, 0);

    MPI_Comm comm;
    comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    vector<Eigen::MatrixXd> res_i(np);
    vector<vector<vector<double>> > res(np);
    int ns, nsol, nsys, nt;
    MPI_Request req[np];

    for (int i = 0; i < np; ++i) {
      int my_worker_id = torsten::mpi::my_worker(i, np, size);
      Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
      ns   = ode.ns();
      nsol = ode.n_sol();
      nsys = ode.n_sys();
      nt   = ode.ts().size();
      res_i[i].resize(nt, nsys);
      if(rank == my_worker_id) {
        res_i[i] = solver.integrate<Ode, false>(ode);
      }
      MPI_Ibcast(res_i[i].data(), res_i[i].size(), MPI_DOUBLE, my_worker_id, comm, &req[i]);
    }

    int finished = 0;
    int flag = 0;
    int index;
    while(finished != np) {
      MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
      if(flag) {
        finished++;
      }
    }

    return res_i;
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
  std::vector<Eigen::Matrix<typename stan::return_type<Tt, T_initial, T_param>::type, // NOLINT
                            Eigen::Dynamic, Eigen::Dynamic> >
  pk_integrate_ode_adams(const F& f,
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
    using Eigen::Matrix;
    using Eigen::MatrixXd;
    using Eigen::Dynamic;
    using Ode = PKCvodesFwdSystem<F, Tt, T_initial, T_param, CV_ADAMS, AD>;
    const int m = theta[0].size();
    const int n = y0[0].size();
    const int np = theta.size(); // population size

    PKCvodesService<typename Ode::Ode> serv(n, m);
    PKCvodesIntegrator solver(rtol, atol, max_num_step);
    
    // make sure MPI is on
    int intialized;
    MPI_Initialized(&intialized);
    stan::math::check_greater("pk_integrate_ode_bdf", "MPI_Intialized", intialized, 0);

    MPI_Comm comm;
    comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    using scalar_type = typename stan::return_type<Tt, T_initial, T_param>::type;

    vector<MatrixXd> res_i(np);
    vector<Matrix<scalar_type, Dynamic, Dynamic> > res(np);
    vector<scalar_type> vars;
    std::vector<double> g;
    int ns, nsol, nsys, nt;
    MPI_Request req[np];

    for (int i = 0; i < np; ++i) {
      int my_worker_id = torsten::mpi::my_worker(i, np, size);
      Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
      vars = ode.vars();
      ns   = ode.ns();
      nsys = ode.n_sys();
      nt   = ode.ts().size();
      nsol = ode.n_sol();
      res_i[i].resize(nt, nsys);
      if(rank == my_worker_id) {
        res_i[i] = solver.integrate<Ode, false>(ode);
      }
      MPI_Ibcast(res_i[i].data(), res_i[i].size(), MPI_DOUBLE, my_worker_id, comm, &req[i]);
      g.resize(ns);
      res[i].resize(nt, n);
      if(rank == my_worker_id) {
        for (int j = 0 ; j < nt; ++j) {
          for (int k = 0; k < n; ++k) {
            for (int l = 0 ; l < ns; ++l) g[l] = res_i[i](j, k * nsol + l + 1);
            res[i](j, k) = precomputed_gradients(res_i[i](j, k * nsol), vars, g);
          }
        }
      }
    }

    int finished = 0;
    int flag = 0;
    int index;
    while(finished != np) {
      MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
      if(flag) {
        int i = index;
        int my_worker_id = torsten::mpi::my_worker(i, np, size);
        Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
        vars = ode.vars();
        if(rank != my_worker_id) {
          for (int j = 0 ; j < nt; ++j) {
            for (int k = 0; k < n; ++k) {
              for (int l = 0 ; l < ns; ++l) g[l] = res_i[i](j, k * nsol + l + 1);
              res[i](j, k) = precomputed_gradients(res_i[i](j, k * nsol), vars, g);
            }
          }
        }
        finished++;
      }
    }
    return res;
  }
#endif

}
}
#endif
