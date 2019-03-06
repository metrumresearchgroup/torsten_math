#ifndef STAN_MATH_TORSTEN_DSOLVE_PK_POPULATION_INTEGRATOR_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PK_POPULATION_INTEGRATOR_HPP

#include <stan/math/torsten/dsolve/pk_cvodes_fwd_system.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_integrator.hpp>
#include <stan/math/torsten/mpi.hpp>
#include <boost/mpi.hpp>

namespace torsten {
  namespace mpi {

    /*
     * For MPI integrator the workflow is to solve at
     * designated rank and pass the results to the rest, while
     * waiting for the results from the other ranks to be
     * passed over. The results are in the form of data so
     * they can be sent over through MPI before assembled
     * into @c var at each rank. The data-only scenario is treated
     * separately as it doesn't require @c var generation.
     * Note that when one rank fails to solve the ODE and
     * throws an exception we need make sure the rest ranks
     * also throw to avoid locking, for that purpose an
     * invalid value is filled into the data passed to the rest
     * ranks so they can detect the exception.
     */
    template <typename F, int Lmm>
    struct PkPopulationIntegrator {

      torsten::dsolve::PKCvodesIntegrator& solver;
      static constexpr double invalid_res_d = -123456789987654321.0;

      PkPopulationIntegrator(torsten::dsolve::PKCvodesIntegrator& solver0) : solver(solver0)
      {}

#ifdef TORSTEN_MPI
      /*
       * MPI solution when the parameters contain @c var
       */ 
      template <typename Tt, typename T_initial, typename T_param,
                typename std::enable_if_t<stan::is_var<typename stan::return_type<Tt, T_initial, T_param>::type>::value >* = nullptr> // NOLINT
      inline
      std::vector<Eigen::Matrix<typename stan::return_type<Tt, T_initial, T_param>::type, // NOLINT
                                Eigen::Dynamic, Eigen::Dynamic> >
      operator()(const F& f,
                 const std::vector<std::vector<T_initial> >& y0,
                 double t0,
                 const std::vector<std::vector<Tt> >& ts,
                 const std::vector<std::vector<T_param> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 std::ostream* msgs) {
        using std::vector;
        using torsten::dsolve::PKCvodesFwdSystem;
        using torsten::dsolve::PKCvodesIntegrator;
        using torsten::PkCvodesSensMethod;
        using Eigen::Matrix;
        using Eigen::MatrixXd;
        using Eigen::Dynamic;
        using Ode = PKCvodesFwdSystem<F, Tt, T_initial, T_param, Lmm, AD>;
        const int m = theta[0].size();
        const int n = y0[0].size();
        const int np = theta.size(); // population size

        torsten::dsolve::PKCvodesService<typename Ode::Ode> serv(n, m);
    
        torsten::mpi::init();

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

        bool is_invalid = false;
        std::ostringstream rank_fail_msg;

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
            try {
              res_i[i] = solver.integrate<Ode, false>(ode);
            } catch (const std::exception& e) {
              is_invalid = true;
              res_i[i].setConstant(invalid_res_d);
              rank_fail_msg << "Rank " << rank << " failed solve id " << i << ": " << e.what();
            }
          }
          MPI_Ibcast(res_i[i].data(), res_i[i].size(), MPI_DOUBLE, my_worker_id, comm, &req[i]);
          g.resize(ns);
          res[i].resize(nt, n);
          if(rank == my_worker_id && !is_invalid) {
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
            finished++;
            if(is_invalid) continue;
            int i = index;
            if (res_i[i].isApproxToConstant(invalid_res_d)) {
              is_invalid = true;
              rank_fail_msg << "Rank " << rank << " received invalid data for id " << i;
            } else {
              if (rank != torsten::mpi::my_worker(i, np, size)) {
                Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
                vars = ode.vars();
                for (int j = 0 ; j < nt; ++j) {
                  for (int k = 0; k < n; ++k) {
                    for (int l = 0 ; l < ns; ++l) g[l] = res_i[i](j, k * nsol + l + 1);
                    res[i](j, k) = precomputed_gradients(res_i[i](j, k * nsol), vars, g);
                  }
                }
              }
            }
          }
        }
        
        if(is_invalid) {
          throw std::runtime_error(rank_fail_msg.str());
        }

        return res;
      }

      /*
       * For data-only MPI solution, we simply pass the
       * results over.
       */
      inline
      std::vector<Eigen::MatrixXd>
      operator()(const F& f,
                 const std::vector<std::vector<double> >& y0,
                 double t0,
                 const std::vector<std::vector<double> >& ts,
                 const std::vector<std::vector<double> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 std::ostream* msgs) {
        using stan::math::var;
        using std::vector;
        using torsten::dsolve::PKCvodesFwdSystem;
        using torsten::dsolve::PKCvodesIntegrator;
        using torsten::PkCvodesSensMethod;
        using Ode = PKCvodesFwdSystem<F, double, double, double, Lmm, AD>;
        const int m = theta[0].size();
        const int n = y0[0].size();
        const int np = theta.size(); // population size

        torsten::dsolve::PKCvodesService<typename Ode::Ode> serv(n, m);
    
        torsten::mpi::init();

        MPI_Comm comm;
        comm = MPI_COMM_WORLD;
        int rank, size;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);

        vector<Eigen::MatrixXd> res_i(np);
        vector<vector<vector<double>> > res(np);
        int nsys, nt;
        MPI_Request req[np];

        bool is_invalid = false;
        std::ostringstream rank_fail_msg;

        for (int i = 0; i < np; ++i) {
          int my_worker_id = torsten::mpi::my_worker(i, np, size);
          Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
          nsys = ode.n_sys();
          nt   = ode.ts().size();
          res_i[i].resize(nt, nsys);
          if(rank == my_worker_id) {
            try {
              res_i[i] = solver.integrate<Ode, false>(ode);
            } catch (const std::exception& e) {
              is_invalid = true;
              res_i[i].setConstant(invalid_res_d);
              rank_fail_msg << "Rank " << rank << " failed solve id " << i << ": " << e.what();
            }
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
            if(is_invalid) continue;
            int i = index;
            if (res_i[i].isApproxToConstant(invalid_res_d)) {
              is_invalid = true;
              rank_fail_msg << "Rank " << rank << " received invalid data for id " << i;
            }        
          }
        }

        if(is_invalid) {
          throw std::runtime_error(rank_fail_msg.str());
        }

        return res_i;
      }
#else
      template <typename Tt, typename T_initial, typename T_param,
                typename std::enable_if_t<stan::is_var<typename stan::return_type<Tt, T_initial, T_param>::type>::value >* = nullptr> // NOLINT
      inline
      std::vector<Eigen::Matrix<typename stan::return_type<Tt, T_initial, T_param>::type, // NOLINT
                                Eigen::Dynamic, Eigen::Dynamic> >
      operator()(const F& f,
                 const std::vector<std::vector<T_initial> >& y0,
                 double t0,
                 const std::vector<std::vector<Tt> >& ts,
                 const std::vector<std::vector<T_param> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 std::ostream* msgs) {
        using std::vector;
        using torsten::dsolve::PKCvodesFwdSystem;
        using torsten::dsolve::PKCvodesIntegrator;
        using torsten::PkCvodesSensMethod;
        using Eigen::Matrix;
        using Eigen::MatrixXd;
        using Eigen::Dynamic;
        using Ode = PKCvodesFwdSystem<F, Tt, T_initial, T_param, Lmm, AD>;
        const int m = theta[0].size();
        const int n = y0[0].size();
        const int np = theta.size(); // population size

        static bool has_warning = false;
        if (!has_warning) {
          std::cout << "Torsten Population ODE solver " << "running sequentially" << "\n";
          has_warning = true;
        }

        static torsten::dsolve::PKCvodesService<typename Ode::Ode> serv(n, m);

        using scalar_type = typename stan::return_type<Tt, T_initial, T_param>::type;
        vector<Matrix<scalar_type, Dynamic, Dynamic> > res(np);

        for (int i = 0; i < np; ++i) {
          Ode ode{serv, f, t0, ts[i], y0[i], theta[i], x_r[i], x_i[i], msgs};
          res[i] = solver.integrate(ode);
        }

        return res;
      }
#endif
    };

    template <typename F, int Lmm>
    constexpr double PkPopulationIntegrator<F, Lmm>::invalid_res_d;

  }
}

#endif
