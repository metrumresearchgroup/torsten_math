#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_POPULATION_INTEGRATOR_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_POPULATION_INTEGRATOR_HPP

#include <stan/math/prim/mat/fun/to_matrix.hpp>
#include <stan/math/torsten/dsolve/pmx_cvodes_fwd_system.hpp>
#include <stan/math/torsten/dsolve/pmx_cvodes_integrator.hpp>
#include <stan/math/torsten/mpi/session.hpp>
#include <stan/math/torsten/mpi/my_worker.hpp>
#include <stan/math/torsten/return_type.hpp>
#include <stan/math/torsten/is_var.hpp>
#include <algorithm>
#include <vector>

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
    struct PMXPopulationIntegrator {

      torsten::dsolve::PMXCvodesIntegrator& solver;
      static constexpr double invalid_res_d = std::numeric_limits<double>::quiet_NaN();

      PMXPopulationIntegrator(torsten::dsolve::PMXCvodesIntegrator& solver0) : solver(solver0)
      {}

#ifdef TORSTEN_MPI
      /*
       * MPI solution when the parameters contain @c var
       */ 
      template <typename Tt, typename T_initial, typename T_param,
                typename std::enable_if_t<torsten::is_var<Tt, T_initial, T_param>::value >* = nullptr> // NOLINT
      inline
      Eigen::Matrix<typename torsten::return_t<Tt, T_initial, T_param>::type, // NOLINT
                    Eigen::Dynamic, Eigen::Dynamic>
      operator()(const F& f,
                 const std::vector<std::vector<T_initial> >& y0,
                 double t0,
                 const std::vector<int>& len,
                 const std::vector<Tt>& ts,
                 const std::vector<std::vector<T_param> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 std::ostream* msgs) {
        using std::vector;
        using torsten::dsolve::PMXCvodesFwdSystem;
        using torsten::dsolve::PMXCvodesIntegrator;
        using torsten::PMXCvodesSensMethod;
        using Eigen::Matrix;
        using Eigen::MatrixXd;
        using Eigen::Dynamic;
        using Ode = PMXCvodesFwdSystem<F, Tt, T_initial, T_param, Lmm, AD>;
        const int m = theta[0].size();
        const int n = y0[0].size();
        const int np = theta.size(); // population size

        torsten::dsolve::PMXCvodesService<typename Ode::Ode> serv(n, m);
    
        MPI_Comm comm = torsten::mpi::Session<NUM_TORSTEN_COMM>::comms[TORSTEN_COMM_ODE_PARM].comm;
        int rank = torsten::mpi::Session<NUM_TORSTEN_COMM>::comms[TORSTEN_COMM_ODE_PARM].rank;
        int size = torsten::mpi::Session<NUM_TORSTEN_COMM>::comms[TORSTEN_COMM_ODE_PARM].size;

        using scalar_type = typename torsten::return_t<Tt, T_initial, T_param>::type;

        vector<MatrixXd> res_d(np);
        Matrix<scalar_type, -1, -1> res(n, ts.size());
        res.setZero();
        vector<scalar_type> vars;
        std::vector<double> g;
        int ns, nsol, nsys, nt;
        std::vector<MPI_Request> req(np);

        bool is_invalid = false;
        std::ostringstream rank_fail_msg;

        // n_req will be updated to contain the number of
        // active requests. These are the requests that we
        // need to make sure to finish.
        int n_req = np;

        typename std::vector<Tt>::const_iterator iter = ts.begin();
        for (int i = 0; i < np; ++i) {
          std::vector<Tt> ts_i(iter, iter + len[i]);
          iter += len[i];
          int my_worker_id = torsten::mpi::my_worker(i, np, size);
          try {
            Ode ode{serv, f, t0, ts_i, y0[i], theta[i], x_r[i], x_i[i], msgs};
            vars = ode.vars();
            ns   = ode.ns();
            nsys = ode.n_sys();
            nt   = ode.ts().size();
            nsol = ode.n_sol();
            res_d[i].resize(nsys, nt);
            res_d[i].setConstant(0.0);

            // success in creating ODE, solve it
            if(rank == my_worker_id) {
              try {
                res_d[i] = solver.integrate<Ode, false>(ode);
              } catch (const std::exception& e) {
                is_invalid = true;
                res_d[i].setConstant(invalid_res_d);
                rank_fail_msg << "Rank " << rank << " failed to solve ODEs for id " << i << ": " << e.what();
              }
            }
          } catch (const std::exception& e) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " failed to create ODEs for id " << i << ": " << e.what();
            n_req = i;
            break;
          }

          MPI_Ibcast(res_d[i].data(), res_d[i].size(), MPI_DOUBLE, my_worker_id, comm, &req[i]);
        }

        int begin_id = 0;
        iter = ts.begin();
        for (int i = 0; i < n_req; ++i) {
          std::vector<Tt> ts_i(iter, iter + len[i]);
          iter += len[i];
          MPI_Wait(&req[i], MPI_STATUS_IGNORE);
          if(is_invalid) continue;
          if (std::isnan(res_d[i](0))) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " received invalid data for id " << i;
          } else {
            Ode ode{serv, f, t0, ts_i, y0[i], theta[i], x_r[i], x_i[i], msgs};
            g.resize(ode.ns());
            vars = ode.vars();
            for (int j = 0 ; j < nt; ++j) {
              for (int k = 0; k < n; ++k) {
                for (int l = 0 ; l < ns; ++l) g[l] = res_d[i](k * nsol + l + 1, j);
                Matrix<scalar_type, -1, -1>::Map(&res(begin_id), n, len[i])(k, j) = precomputed_gradients(res_d[i](k * nsol, j), vars, g);
                // res[i](k, j) = precomputed_gradients(res_d[i](k * nsol, j), vars, g);
              }
            }
          }
          begin_id += len[i] * n;
        }
        
        if(is_invalid) {
          throw std::runtime_error(rank_fail_msg.str());
        }

        return res;
      }

      /*
       * For data-only MPI solution, we simply pass the
       * results over.
       * @param len num of time steps for each subject.
       * @param ts time steps for the entire group.
       */
      inline Eigen::MatrixXd operator()(const F& f,
                                        const std::vector<std::vector<double> >& y0,    // NOLINT
                                        double t0,
                                        const std::vector<int>& len,
                                        const std::vector<double>& ts,
                                        const std::vector<std::vector<double> >& theta, // NOLINT
                                        const std::vector<std::vector<double> >& x_r,   // NOLINT
                                        const std::vector<std::vector<int> >& x_i,      // NOLINT
                                        std::ostream* msgs) {
        using stan::math::var;
        using std::vector;
        using Eigen::Matrix;
        using Eigen::MatrixXd;
        using Eigen::Dynamic;
        using torsten::dsolve::PMXCvodesFwdSystem;
        using torsten::dsolve::PMXCvodesIntegrator;
        using torsten::PMXCvodesSensMethod;
        using Ode = PMXCvodesFwdSystem<F, double, double, double, Lmm, AD>;
        const int m = theta[0].size();
        const int n = y0[0].size();
        const int np = theta.size(); // population size

        torsten::dsolve::PMXCvodesService<typename Ode::Ode> serv(n, m);
    
        MPI_Comm comm = torsten::mpi::Session<NUM_TORSTEN_COMM>::comms[TORSTEN_COMM_ODE_DATA].comm;
        int rank = torsten::mpi::Session<NUM_TORSTEN_COMM>::comms[TORSTEN_COMM_ODE_DATA].rank;
        int size = torsten::mpi::Session<NUM_TORSTEN_COMM>::comms[TORSTEN_COMM_ODE_DATA].size;

        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n, ts.size());
        std::vector<MPI_Request> req(np);
        std::vector<int> begin_ids(np);

        bool is_invalid = false;
        int n_req = np;
        std::ostringstream rank_fail_msg;

        std::vector<double>::const_iterator iter = ts.begin();
        int begin_id = 0;
        for (int i = 0; i < np; ++i) {
          std::vector<double> ts_i(iter, iter + len[i]);
          iter += len[i];
          int size_id = len[i] * n;
          int my_worker_id = torsten::mpi::my_worker(i, np, size);
          try {
            Ode ode{serv, f, t0, ts_i, y0[i], theta[i], x_r[i], x_i[i], msgs};

            // success in creating ODE, solve it
            if(rank == my_worker_id) {
              try {
                MatrixXd::Map(&res(begin_id), n, len[i]) = solver.integrate<Ode, false>(ode);
              } catch (const std::exception& e) {
                is_invalid = true;
                MatrixXd::Map(&res(begin_id), n, len[i]).setConstant(invalid_res_d);
                rank_fail_msg << "Rank " << rank << " failed to solve ODEs for id " << i << ": " << e.what();
              }
            }
          } catch (const std::exception& e) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " failed to create ODEs for id " << i << ": " << e.what();
            n_req = i;
            break;
          }

          MPI_Ibcast(&res(begin_id), size_id, MPI_DOUBLE, my_worker_id, comm, &req[i]);
          begin_ids[i] = begin_id;
          begin_id += size_id;
        }

        int finished = 0;
        int index;
        while(finished != n_req) {
          MPI_Waitany(n_req, req.data(), &index, MPI_STATUS_IGNORE);
          finished++;
          if(is_invalid) continue;
          int i = index;
          if (std::isnan(res(begin_ids[i]))) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " received invalid data for id " << i;
          }
        }

        if(is_invalid) {
          throw std::runtime_error(rank_fail_msg.str());
        }

        return res;
      }
#else
      template <typename Tt, typename T_initial, typename T_param>
      inline Eigen::Matrix<typename torsten::return_t<Tt, T_initial, T_param>::type, // NOLINT
                           Eigen::Dynamic, Eigen::Dynamic>
      operator()(const F& f,
                 const std::vector<std::vector<T_initial> >& y0,
                 double t0,
                 const std::vector<int>& len,
                 const std::vector<Tt>& ts,
                 const std::vector<std::vector<T_param> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 std::ostream* msgs) {
        using std::vector;
        using torsten::dsolve::PMXCvodesFwdSystem;
        using torsten::dsolve::PMXCvodesIntegrator;
        using torsten::PMXCvodesSensMethod;
        using Eigen::Matrix;
        using Eigen::MatrixXd;
        using Eigen::Dynamic;
        using Ode = PMXCvodesFwdSystem<F, Tt, T_initial, T_param, Lmm, AD>;
        const int m = theta[0].size();
        const int n = y0[0].size();
        const int np = theta.size(); // population size

        static bool has_warning = false;
        if (!has_warning) {
          std::cout << "Torsten Population ODE solver " << "running sequentially" << "\n";
          has_warning = true;
        }

        static torsten::dsolve::PMXCvodesService<typename Ode::Ode> serv(n, m);

        using scalar_type = typename torsten::return_t<Tt, T_initial, T_param>::type;
        Matrix<scalar_type, Dynamic, Dynamic> res(n, ts.size());

        vector<vector<scalar_type> > res_i;
        typename std::vector<Tt>::const_iterator iter = ts.begin();
        int begin_id = 0;
        for (int i = 0; i < np; ++i) {
          std::vector<Tt> ts_i(iter, iter + len[i]);
          iter += len[i];
          Ode ode{serv, f, t0, ts_i, y0[i], theta[i], x_r[i], x_i[i], msgs};
          Matrix<scalar_type, -1, -1>::Map(&res(begin_id), n, len[i])
            = stan::math::to_matrix(solver.integrate(ode)).transpose();
          begin_id += n * len[i];
        }

        return res;
      }
#endif
    };

    template <typename F, int Lmm>
    constexpr double PMXPopulationIntegrator<F, Lmm>::invalid_res_d;

  }
}

#endif
