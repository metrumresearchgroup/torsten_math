#ifndef STAN_MATH_TORSTEN_PKMODEL_REFACTOR_PRED_HPP
#define STAN_MATH_TORSTEN_PKMODEL_REFACTOR_PRED_HPP

#include <Eigen/Dense>
#include <vector>
#include <stan/math/torsten/dsolve/pk_vars.hpp>
#include <stan/math/torsten/mpi/precomputed_gradients.hpp>
#include <stan/math/torsten/duplicate.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_linode_model.hpp>


namespace torsten{
  /*
   * the wrapper is aware of @c T_model so it build model
   * accordingly.
   */
  template<typename T_model, typename... T_pred>
  struct PredWrapper{
    /**
     * Every Torsten function calls Pred.
     *
     * Predicts the amount in each compartment for each event,
     * given the event schedule and the parameters of the model.
     *
     * Proceeds in two steps. First, computes all the events that
     * are not included in the original data set but during which
     * amounts in the system get updated. Secondly, predicts
     * the amounts in each compartment sequentially by going
     * through the augmented schedule of events. The returned pred
     * Matrix only contains the amounts in the event originally
     * specified by the users.
     *
     * This function is valid for all models. What changes from one
     * model to the other are the Pred1 and PredSS functions, which
     * calculate the amount at an individual event.
     *
     * @tparam T_em type the @c EventsManager
     * @param[in] time times of events
     * @param[in] amt amount at each event
     * @param[in] rate rate at each event
     * @param[in] ii inter-dose interval at each event
     * @param[in] evid event identity:
     *                    (0) observation
     *                    (1) dosing
     *                    (2) other
     *                    (3) reset
     *                    (4) reset AND dosing
     * @param[in] cmt compartment number at each event (starts at 1)
     * @param[in] addl additional dosing at each event
     * @param[in] ss steady state approximation at each event
     * (0: no, 1: yes)
     * @param[in] pMatrix parameters at each event
     * @param[in] addParm additional parameters at each event
     * @parem[in] model basic info for ODE model and evolution operators
     * @param[in] SystemODE matrix describing linear ODE system that
     * defines compartment model. Used for matrix exponential solutions.
     * Included because it may get updated in modelParameters.
     * @return a matrix with predicted amount in each compartment
     * at each event.
     */
    template<typename... T_em, typename... Ts>
    static void pred(const EventsManager<T_em...>& em,
                     Eigen::Matrix<typename EventsManager<T_em...>::T_scalar, -1, -1>& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      using::stan::math::multiply;
      using refactor::PKRec;

      using scalar = typename EventsManager<T_em...>::T_scalar;

      auto events = em.events();
      auto model_rate = em.rates();
      auto model_amt = em.amts();
      auto model_par = em.pars();
      res.resize(em.nKeep, em.ncmt);
      PKRec<scalar> zeros = PKRec<scalar>::Zero(em.ncmt);
      PKRec<scalar> init = zeros;
      auto dt = events.time(0);
      auto tprev = events.time(0);
      Matrix<scalar, Dynamic, 1> pred1;
      int ikeep = 0;

      for (size_t i = 0; i < events.size(); i++) {
        if (events.is_reset(i)) {
          dt = 0;
          init = zeros;
        } else if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
          decltype(tprev) model_time = events.time(i); // FIXME: time is not t0 but for adjust within SS solver
          // auto model_par = parameter.get_RealParameters();
          // FIX ME: we need a better way to relate model type to parameter type
          T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
          pred1 = multiply(pkmodel.solve(model_amt[i], //NOLINT
                                         events.rate(i),
                                         events.ii(i),
                                         events.cmt(i),
                                         pred_pars...),
                           scalar(1.0));

          // the object PredSS returns doesn't always have a scalar type. For
          // instance, PredSS does not depend on tlag, but pred does. So if
          // tlag were a var, the code must promote PredSS to match the type
          // of pred1. This is done by multiplying predSS by a Scalar.

          if (events.ss(i) == 2)
            init += pred1;  // steady state without reset
          else
            init = pred1;  // steady state with reset (ss = 1)
        } else {           // non-steady dosing event
          dt = events.time(i) - tprev;
          decltype(tprev) model_time = tprev;

          // FIX ME: we need a better way to relate model type to parameter type
          T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};

          pred1 = pkmodel.solve(dt, pred_pars...);
          init = pred1;
        }

        if (events.is_bolus_dosing(i)) {
          init(0, events.cmt(i) - 1) += model_amt[i];
        }

        if (events.keep(i)) {
          res.row(ikeep) = init;
          ikeep++;
        }
        tprev = events.time(i);
      }
    }

#ifdef TORSTEN_MPI
    /*
     * MPI solution when @c amt is data
     */
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename... Ts,
             typename std::enable_if_t<!stan::is_var<T1>::value >* = nullptr>
    static void pred(int nCmt,
                     const std::vector<std::vector<T0> >& time,
                     const std::vector<std::vector<T1> >& amt,
                     const std::vector<std::vector<T2> >& rate,
                     const std::vector<std::vector<T3> >& ii,
                     const std::vector<std::vector<int> >& evid,
                     const std::vector<std::vector<int> >& cmt,
                     const std::vector<std::vector<int> >& addl,
                     const std::vector<std::vector<int> >& ss,
                     const std::vector<std::vector<std::vector<T4> > >& pMatrix,
                     const std::vector<std::vector<std::vector<T5> > >& biovar,
                     const std::vector<std::vector<std::vector<T6> > >& tlag,
                     std::vector<Eigen::Matrix<typename EventsManager<T0, T1, T2, T3, T4, T5, T6>::T_scalar, -1, -1> >& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using Eigen::Matrix;
      using Eigen::MatrixXd;
      using Eigen::VectorXd;
      using Eigen::Dynamic;
      using std::vector;
      using::stan::math::var;
      using::stan::math::multiply;
      using refactor::PKRec;

      using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;
      using scalar = typename EM::T_scalar;

      const int np = time.size();

      // make sure MPI is on
      int intialized;
      MPI_Initialized(&intialized);
      stan::math::check_greater("PredWrapper::pred", "MPI_Intialized", intialized, 0);

      MPI_Comm comm;
      comm = MPI_COMM_WORLD;
      int rank, size;
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);

      MPI_Request req[np];
      vector<MatrixXd> res_d(np);
      
      res.resize(np);
      std::vector<int> nvars(np);

      for (int id = 0; id < np; ++id) {

        /* For every rank */

        EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
        res[id].resize(em.nKeep, em.ncmt);

        auto events = em.events();
        auto model_rate = em.rates();
        auto model_amt = em.amts();
        auto model_par = em.pars();
        PKRec<scalar> zeros = PKRec<scalar>::Zero(em.ncmt);
        PKRec<scalar> init = zeros;

        nvars[id] = T_model::nvars(events.time(0), init, model_rate[0], model_par[0]);
        int nsys = T_model::n_sys(events.time(0), init, model_rate[0], model_par[0]);
        res_d[id].resize(events.size(), nsys);

        typename EM::T_time dt = events.time(0);
        typename EM::T_time tprev = events.time(0);

        PKRec<double> pred1 = VectorXd::Zero(nsys);
        int ikeep = 0;

        int my_worker_id = torsten::mpi::my_worker(id, np, size);

        /* only solver rank */

        if (rank == my_worker_id) {
          std::cout << "rank " << rank << " solving " << id << "\n";

          for (size_t i = 0; i < events.size(); i++) {
            if (events.is_reset(i)) {
              dt = 0;
              init = zeros;
              pred1 = Eigen::VectorXd::Zero(nsys);
            } else if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
              decltype(tprev) model_time = events.time(i); // FIXME: time is not t0 but for adjust within SS solver
              // auto model_par = parameter.get_RealParameters();
              T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
              vector<var> v_i = pkmodel.vars(model_amt[i], events.rate(i), events.ii(i));
              pred1 = pkmodel.solve_d(model_amt[i], events.rate(i), events.ii(i), events.cmt(i), pred_pars...);
              if (events.ss(i) == 2)
                init += torsten::mpi::precomputed_gradients(pred1, v_i);  // steady state without reset
              else
                init = torsten::mpi::precomputed_gradients(pred1, v_i);  // steady state with reset (ss = 1)
            } else {
              dt = events.time(i) - tprev;
              if (dt > 0) {
                typename EM::T_time model_time = tprev;
                T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
                vector<var> v_i = pkmodel.vars(events.time(i));
                pred1 = pkmodel.solve_d(dt, pred_pars...);
                init = torsten::mpi::precomputed_gradients(pred1, v_i);
              }
            }

            if (events.is_bolus_dosing(i)) {
              init(0, events.cmt(i) - 1) += model_amt[i];
              pred1((events.cmt(i) - 1) * (nvars[id] + 1)) += model_amt[i];
            }

            // we need every step, not just ikeep steps.
            res_d[id].row(i) = pred1;

            if (events.keep(i)) {
              res[id].row(ikeep) = init;
              ikeep++;
            }
            tprev = events.time(i);
          }
        }

        MPI_Ibcast(res_d[id].data(), res_d[id].size(), MPI_DOUBLE, my_worker_id, comm, &req[id]);
      }

      /* Assemble other ranks' results */

      int finished = 0;
      int flag = 0;
      int index;
      while (finished != np) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          int id = index;
          int my_worker_id = torsten::mpi::my_worker(id, np, size);
          if (rank != my_worker_id) {
            std::cout << "rank " << rank << " syncing " << id << "\n";

            EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
            res[id].resize(em.nKeep, em.ncmt);
            auto events = em.events();
            auto model_rate = em.rates();
            auto model_amt = em.amts();
            auto model_par = em.pars();
            PKRec<scalar> zeros = PKRec<scalar>::Zero(em.ncmt);
            PKRec<scalar> init = zeros;
            nvars[id] = T_model::nvars(events.time(0), init, model_rate[0], model_par[0]);
            int nsys = T_model::n_sys(events.time(0), init, model_rate[0], model_par[0]);
            typename EM::T_time dt = events.time(0);
            typename EM::T_time tprev = events.time(0);
            PKRec<double> pred1 = VectorXd::Zero(nsys);
            int ikeep = 0;

            for (size_t i = 0; i < events.size(); i++) {
              if (events.is_reset(i)) {
                dt = 0;
                init = zeros;
              } else if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
                typename EM::T_time model_time = events.time(i); // FIXME: time is not t0 but for adjust within SS solver
                T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
                vector<var> v_i = pkmodel.vars(model_amt[i], events.rate(i), events.ii(i));
                pred1 = res_d[id].row(i);
                if (events.ss(i) == 2)
                  init += torsten::mpi::precomputed_gradients(pred1, v_i);  // steady state without reset
                else
                  init = torsten::mpi::precomputed_gradients(pred1, v_i);  // steady state with reset (ss = 1)
              } else {
                dt = events.time(i) - tprev;
                if (dt > 0) {
                  typename EM::T_time model_time = tprev;
                  T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
                  vector<var> v_i = pkmodel.vars(events.time(i));
                  pred1 = res_d[id].row(i);
                  init = torsten::mpi::precomputed_gradients(pred1, v_i);
                }
              }

              if (events.is_bolus_dosing(i)) {
                init(0, events.cmt(i) - 1) += model_amt[i];
              }

              if (events.keep(i)) {
                res[id].row(ikeep) = init;
                ikeep++;
              }
              tprev = events.time(i);
            }
          }
          finished++;
        }
      }
      // finished
    }

    /*
     * MPI, data-only version
     */
    template<typename... Ts>
    static void pred(int nCmt,
                     const std::vector<std::vector<double> >& time,
                     const std::vector<std::vector<double> >& amt,
                     const std::vector<std::vector<double> >& rate,
                     const std::vector<std::vector<double> >& ii,
                     const std::vector<std::vector<int> >& evid,
                     const std::vector<std::vector<int> >& cmt,
                     const std::vector<std::vector<int> >& addl,
                     const std::vector<std::vector<int> >& ss,
                     const std::vector<std::vector<std::vector<double> > >& pMatrix,
                     const std::vector<std::vector<std::vector<double> > >& biovar,
                     const std::vector<std::vector<std::vector<double> > >& tlag,
                     std::vector<Eigen::Matrix<double, -1, -1> >& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using Eigen::Matrix;
      using Eigen::MatrixXd;
      using Eigen::VectorXd;
      using Eigen::Dynamic;
      using std::vector;
      using::stan::math::var;
      using::stan::math::multiply;
      using refactor::PKRec;

      using EM = EventsManager<double, double, double, double, double, double, double>;
      const int np = time.size();

      // make sure MPI is on
      int intialized;
      MPI_Initialized(&intialized);
      stan::math::check_greater("PredWrapper::pred", "MPI_Intialized", intialized, 0);

      MPI_Comm comm;
      comm = MPI_COMM_WORLD;
      int rank, size;
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);

      MPI_Request req[np];
      vector<MatrixXd> res_d(np);
      
      res.resize(np);
      std::vector<int> nvars(np);

      for (int id = 0; id < np; ++id) {

        /* For every rank */

        EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
        res[id].resize(em.nKeep, em.ncmt);

        auto events = em.events();
        auto model_rate = em.rates();
        auto model_amt = em.amts();
        auto model_par = em.pars();
        PKRec<double> zeros = PKRec<double>::Zero(em.ncmt);
        PKRec<double> init = zeros;

        res_d[id].resize(events.size(), em.ncmt);

        double dt = events.time(0);
        double tprev = events.time(0);

        PKRec<double> pred1 = VectorXd::Zero(em.ncmt);
        int ikeep = 0;

        int my_worker_id = torsten::mpi::my_worker(id, np, size);

        /* only solver rank */

        if (rank == my_worker_id) {
          std::cout << "rank " << rank << " solving " << id << "\n";

          for (size_t i = 0; i < events.size(); i++) {
            if (events.is_reset(i)) {
              dt = 0;
              init = zeros;
              pred1 = Eigen::VectorXd::Zero(em.ncmt);
            } else if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
              decltype(tprev) model_time = events.time(i); // FIXME: time is not t0 but for adjust within SS solver
              T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
              pred1 = pkmodel.solve_d(model_amt[i], events.rate(i), events.ii(i), events.cmt(i), pred_pars...);
              if (events.ss(i) == 2)
                init += pred1;  // steady state without reset
              else
                init = pred1;  // steady state with reset (ss = 1)
            } else {
              dt = events.time(i) - tprev;
              if (dt > 0) {
                double model_time = tprev;
                T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
                pred1 = pkmodel.solve_d(dt, pred_pars...);
                init = pred1;
              }
            }

            if (events.is_bolus_dosing(i)) {
              init(0, events.cmt(i) - 1) += model_amt[i];
              pred1(0, events.cmt(i) - 1) += model_amt[i];
            }

            // we need every step, not just ikeep steps.
            res_d[id].row(i) = pred1;

            if (events.keep(i)) {
              res[id].row(ikeep) = init;
              ikeep++;
            }
            tprev = events.time(i);
          }
        }

        MPI_Ibcast(res_d[id].data(), res_d[id].size(), MPI_DOUBLE, my_worker_id, comm, &req[id]);
      }

      /* Assemble other ranks' results */

      int finished = 0;
      int flag = 0;
      int index;
      while (finished != np) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          int id = index;
          int my_worker_id = torsten::mpi::my_worker(id, np, size);
          if (rank != my_worker_id) {
            std::cout << "rank " << rank << " syncing " << id << "\n";

            EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
            res[id].resize(em.nKeep, em.ncmt);
            auto events = em.events();
            auto model_rate = em.rates();
            auto model_amt = em.amts();
            auto model_par = em.pars();
            PKRec<double> zeros = PKRec<double>::Zero(em.ncmt);
            PKRec<double> init = zeros;
            typename EM::T_time dt = events.time(0);
            typename EM::T_time tprev = events.time(0);
            int ikeep = 0;

            for (size_t i = 0; i < events.size(); i++) {
              if (events.is_reset(i)) {
                dt = 0;
                init = zeros;
              } else if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
                if (events.ss(i) == 2)
                  init += res_d[id].row(i);  // steady state without reset
                else
                  init = res_d[id].row(i);  // steady state with reset (ss = 1)
              } else {
                dt = events.time(i) - tprev;
                if (dt > 0) {
                  init = res_d[id].row(i);
                }
              }

              if (events.is_bolus_dosing(i)) {
                init(0, events.cmt(i) - 1) += model_amt[i];
              }

              if (events.keep(i)) {
                res[id].row(ikeep) = init;
                ikeep++;
              }
              tprev = events.time(i);
            }
          }
          finished++;
        }
      }
      // finished
    }
    
#else
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename... Ts>
    static void pred(int nCmt,
                     const std::vector<std::vector<T0> >& time,
                     const std::vector<std::vector<T1> >& amt,
                     const std::vector<std::vector<T2> >& rate,
                     const std::vector<std::vector<T3> >& ii,
                     const std::vector<std::vector<int> >& evid,
                     const std::vector<std::vector<int> >& cmt,
                     const std::vector<std::vector<int> >& addl,
                     const std::vector<std::vector<int> >& ss,
                     const std::vector<std::vector<std::vector<T4> > >& pMatrix,
                     const std::vector<std::vector<std::vector<T5> > >& biovar,
                     const std::vector<std::vector<std::vector<T6> > >& tlag,
                     std::vector<Eigen::Matrix<typename EventsManager<T0, T1, T2, T3, T4, T5, T6>::T_scalar, -1, -1>>& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;
      const int np = time.size();

      res.resize(np);

      for (int i = 0; i < np; ++i) {
        EM em(nCmt, time[i], amt[i], rate[i], ii[i], evid[i], cmt[i], addl[i], ss[i], pMatrix[i], biovar[i], tlag[i]);
        res[i].resize(em.nKeep, em.ncmt);
        pred(em, res[i], pred_pars..., model_pars...);
      }
    }    
#endif
  };

}

#endif
