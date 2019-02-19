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
  struct PredWrapper {

    /*
     * Data used to fill the results when computation throws exception.
     */
    static constexpr double invalid_res_d = -123456789987654321.0;

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
    template<typename T_em, typename... Ts>
    static void pred(const T_em& em,
                     Eigen::Matrix<typename T_em::T_scalar, -1, -1>& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      using::stan::math::multiply;
      using refactor::PKRec;

      using scalar = typename T_em::T_scalar;

      res.resize(em.nKeep, em.ncmt);
      PKRec<scalar> init(em.ncmt);
      init.setZero();

      try {
        for (int ik = 0; ik < em.nKeep; ik++) {
          int ibegin = ik == 0 ? 0 : em.keep_ev[ik-1] + 1;
          int iend = em.keep_ev[ik] + 1;
          for (int i = ibegin; i < iend; ++i) {
            stepper(i, init, em, pred_pars..., model_pars...);
          }
          res.row(ik) = init;
        }
      } catch (const std::exception& e) {
        throw;
      }
    }

    /*
     * Step through a range of events.
     */
    template<typename T_em, typename... Ts>
    static void stepper(int i, refactor::PKRec<typename T_em::T_scalar>& init,
                        const T_em& em, const T_pred... pred_pars, const Ts... model_pars) {
      auto events = em.events();
      auto model_rate = em.rates();
      auto model_amt = em.amts();
      auto model_par = em.pars();

      using scalar = typename T_em::T_scalar;
      typename T_em::T_time dt;
      typename T_em::T_time tprev = i == 0 ? events.time(0) : events.time(i-1);

      Eigen::Matrix<scalar, -1, 1> pred1;

      if (events.is_reset(i)) {
        dt = 0;
        init.setZero();
      } else if (events.is_ss_dosing(i)) {  // steady state event
        typename T_em::T_time model_time = events.time(i); // FIXME: time is not t0 but for adjust within SS solver
        T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
        pred1 = stan::math::multiply(pkmodel.solve(model_amt[i], //NOLINT
                                                   events.rate(i),
                                                   events.ii(i),
                                                   events.cmt(i),
                                                   pred_pars...),
                                     scalar(1.0));

        if (events.ss(i) == 2)
          init += pred1;  // steady state without reset
        else
          init = pred1;  // steady state with reset (ss = 1)
      } else {           // non-steady dosing event
        dt = events.time(i) - tprev;
        typename T_em::T_time model_time = tprev;

        T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};

        pred1 = pkmodel.solve(dt, pred_pars...);
        init = pred1;
      }

      if (events.is_bolus_dosing(i)) {
        init(0, events.cmt(i) - 1) += model_amt[i];
      }
      tprev = events.time(i);
    }

    template<typename T_em, typename... Ts>
    static void stepper_solve(int i, refactor::PKRec<typename T_em::T_scalar>& init,
                        refactor::PKRec<double>& sol_d,
                        const T_em& em, const T_pred... pred_pars, const Ts... model_pars) {
      using std::vector;
      using stan::math::var;

      auto events = em.events();
      auto model_rate = em.rates();
      auto model_amt = em.amts();
      auto model_par = em.pars();

      typename T_em::T_time dt;
      typename T_em::T_time tprev = i == 0 ? events.time(0) : events.time(i-1);

      if (events.is_reset(i)) {
        dt = 0;
        init.setZero();
      } else if (events.is_ss_dosing(i)) {  // steady state event
        typename T_em::T_time model_time = events.time(i);
        T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
        vector<var> v_i = pkmodel.vars(model_amt[i], events.rate(i), events.ii(i));
        sol_d = pkmodel.solve_d(model_amt[i], events.rate(i), events.ii(i), events.cmt(i), pred_pars...);
        if (events.ss(i) == 2)
          init += torsten::mpi::precomputed_gradients(sol_d, v_i);  // steady state without reset
        else
          init = torsten::mpi::precomputed_gradients(sol_d, v_i);  // steady state with reset (ss = 1)
      } else {
        dt = events.time(i) - tprev;
        if (dt > 0) {
          typename T_em::T_time model_time = tprev;
          T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
          vector<var> v_i = pkmodel.vars(events.time(i));
          sol_d = pkmodel.solve_d(dt, pred_pars...);
          init = torsten::mpi::precomputed_gradients(sol_d, v_i);
        }
      }

      if (events.is_bolus_dosing(i)) {
        init(0, events.cmt(i) - 1) += model_amt[i];
      }
      tprev = events.time(i);
    }

    template<typename T_em, typename... Ts>
    static void stepper_sync(int i, refactor::PKRec<typename T_em::T_scalar>& init,
                        refactor::PKRec<double>& sol_d,
                             const T_em& em, const T_pred... pred_pars, const Ts... model_pars) {
      using std::vector;
      using stan::math::var;

      auto events = em.events();
      auto model_rate = em.rates();
      auto model_amt = em.amts();
      auto model_par = em.pars();

      typename T_em::T_time dt;
      typename T_em::T_time tprev = i == 0 ? events.time(0) : events.time(i-1);

      if (events.is_reset(i)) {
        dt = 0;
        init.setZero();
      } else if (events.is_ss_dosing(i)) {  // steady state event
        typename T_em::T_time model_time = events.time(i);
        T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
        vector<var> v_i = pkmodel.vars(model_amt[i], events.rate(i), events.ii(i));
        int nsys = torsten::pk_nsys(em.ncmt, v_i.size());
        if (events.ss(i) == 2)
          init += torsten::mpi::precomputed_gradients(sol_d.segment(0, nsys), v_i);  // steady state without reset
        else
          init = torsten::mpi::precomputed_gradients(sol_d.segment(0, nsys), v_i);  // steady state with reset (ss = 1)
      } else {
        dt = events.time(i) - tprev;
        if (dt > 0) {
          typename T_em::T_time model_time = tprev;
          T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};
          vector<var> v_i = pkmodel.vars(events.time(i));
          int nsys = torsten::pk_nsys(em.ncmt, v_i.size());
          init = torsten::mpi::precomputed_gradients(sol_d.segment(0, nsys), v_i);
        }
      }

      if (events.is_bolus_dosing(i)) {
        init(0, events.cmt(i) - 1) += model_amt[i];
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

      // const double invalid_res_d = -123456789987654321.0;

      const int np = time.size();
      bool is_invalid = false;
      std::ostringstream rank_fail_msg;

      torsten::mpi::init();

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

        int nKeep = evid[id].size();

        res[id].resize(nKeep, nCmt);

        int nvar = T_model::nvars(nCmt, pMatrix[id][0].size());
        int nvar_ss = T_model::template nvars<typename EM::T_amt, T2, T3>(pMatrix[id][0].size());
        int nev = EM::nevents(time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
        res_d[id].resize(nev, EM::has_ss_dosing(evid[id], ss[id]) ? torsten::pk_nsys(nCmt, nvar, nvar_ss) : torsten::pk_nsys(nCmt, nvar));

        int my_worker_id = torsten::mpi::my_worker(id, np, size);

        /* only solver rank */

        if (rank == my_worker_id) {
          if (is_invalid) {
            res_d[id].setConstant(invalid_res_d);
          } else {
            EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
            auto events = em.events();
            assert(nev == events.size());
            assert(nKeep == em.nKeep);

            PKRec<scalar> init(em.ncmt); init.setZero();
            PKRec<double> pred1;
            int ikeep = 0;

            try {
              for (size_t i = 0; i < events.size(); i++) {
                stepper_solve(i, init, pred1, em, pred_pars..., model_pars...);
                res_d[id].row(i).segment(0, pred1.size()) = pred1;
                if (events.keep(i)) {
                  res[id].row(ikeep) = init;
                  ikeep++;
                }
              }
            } catch (const std::exception& e) {
              is_invalid = true;
              res_d[id].setConstant(invalid_res_d);
              rank_fail_msg << "Rank " << rank << " failed solve id " << id << ": " << e.what();
            }
          }
        }
        MPI_Ibcast(res_d[id].data(), res_d[id].size(), MPI_DOUBLE, my_worker_id, comm, &req[id]);
      }

      int finished = 0;
      int flag = 0;
      int index;
      while (finished != np && size > 1) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          finished++;
          if (is_invalid) continue;
          int id = index;
          if (res_d[id].isApproxToConstant(invalid_res_d)) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " received invalid data for id " << id;
          } else {
            if (rank != torsten::mpi::my_worker(id, np, size)) {
              EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
              PKRec<scalar> init(nCmt); init.setZero();
              PKRec<double> pred1 = VectorXd::Zero(res_d[id].cols());
              int ikeep = 0;
              for (size_t i = 0; i < em.events().size(); i++) {
                pred1 = res_d[id].row(i);
                stepper_sync(i, init, pred1, em, pred_pars..., model_pars...);
                if (em.events().keep(i)) {
                  res[id].row(ikeep) = init;
                  ikeep++;
                }
              }
            }
          }
        }
      }

      // std::cout << "Torsten MPI rank: " << rank << " done" << "\n";
      if(is_invalid) {
        throw std::runtime_error(rank_fail_msg.str());
      }
    }

    /*
     * MPI solution when @c amt is data, and the population
     * information passed in as ragged arrays.
     */
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename... Ts,
             typename std::enable_if_t<!stan::is_var<T1>::value >* = nullptr>
    static void pred(int nCmt,
                     const std::vector<int>& len,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<int>& len_pMatrix,
                     const std::vector<std::vector<T4> >& pMatrix,
                     const std::vector<int>& len_biovar,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<int>& len_tlag,
                     const std::vector<std::vector<T6> >& tlag,
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

      const int np = len.size();
      bool is_invalid = false;
      std::ostringstream rank_fail_msg;

      torsten::mpi::init();

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

      int i0 = 0, i0_pMatrix = 0, i0_biovar = 0, i0_tlag = 0;
      std::vector<int> j0(np), j0_pMatrix(np), j0_biovar(np), j0_tlag(np);
      for (int id = 0; id < np; ++id) {

        /* For every rank */

        j0[id]         = i0;
        j0_pMatrix[id] = i0_pMatrix;
        j0_biovar[id]  = i0_biovar;
        j0_tlag[id]    = i0_tlag;

        int nKeep = len[id];

        res[id].resize(nKeep, nCmt);

        int nvar = T_model::nvars(nCmt, pMatrix[i0_pMatrix].size());
        int nvar_ss = T_model::template nvars<typename EM::T_amt, T2, T3>(pMatrix[i0_pMatrix].size());
        int nev = EM::nevents(i0, len[id], time, amt, rate, ii, evid, cmt, addl, ss,
                              i0_pMatrix, len_pMatrix[id], pMatrix,
                              i0_biovar, len_biovar[id], biovar,
                              i0_tlag, len_tlag[id], tlag);

        // FIXME: has_ss_dosing shouldn't test the entire
        // population but only the individual
        res_d[id].resize(nev, EM::has_ss_dosing(evid, ss) ? torsten::pk_nsys(nCmt, nvar, nvar_ss) : torsten::pk_nsys(nCmt, nvar));

        int my_worker_id = torsten::mpi::my_worker(id, np, size);

        /* only solver rank */

        if (rank == my_worker_id) {
          if (is_invalid) {
            res_d[id].setConstant(invalid_res_d);
          } else {
            EM em(nCmt, i0, len[id], time, amt, rate, ii, evid, cmt, addl, ss,
                  i0_pMatrix, len_pMatrix[id], pMatrix,
                  i0_biovar, len_biovar[id], biovar,
                  i0_tlag, len_tlag[id], tlag);
            auto events = em.events();
            assert(nev == events.size());
            assert(nKeep == em.nKeep);

            PKRec<scalar> init(em.ncmt); init.setZero();
            PKRec<double> pred1;
            int ikeep = 0;

            try {
              for (size_t i = 0; i < events.size(); i++) {
                stepper_solve(i, init, pred1, em, pred_pars..., model_pars...);
                res_d[id].row(i).segment(0, pred1.size()) = pred1;
                if (events.keep(i)) {
                  res[id].row(ikeep) = init;
                  ikeep++;
                }
              }
            } catch (const std::exception& e) {
              is_invalid = true;
              res_d[id].setConstant(invalid_res_d);
              rank_fail_msg << "Rank " << rank << " failed solve id " << id << ": " << e.what();
            }
          }
        }
        MPI_Ibcast(res_d[id].data(), res_d[id].size(), MPI_DOUBLE, my_worker_id, comm, &req[id]);

        i0         += len[id];
        i0_pMatrix += len_pMatrix[id];
        i0_biovar  += len_biovar[id];
        i0_tlag    += len_tlag[id];
      }

      int finished = 0;
      int flag = 0;
      int index;
      while (finished != np && size > 1) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          finished++;
          if (is_invalid) continue;
          int id = index;
          if (res_d[id].isApproxToConstant(invalid_res_d)) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " received invalid data for id " << id;
          } else {
            if (rank != torsten::mpi::my_worker(id, np, size)) {
              i0         = j0[id];
              i0_pMatrix = j0_pMatrix[id];
              i0_biovar  = j0_biovar[id];
              i0_tlag    = j0_tlag[id];
              EM em(nCmt, i0, len[id], time, amt, rate, ii, evid, cmt, addl, ss,
                    i0_pMatrix, len_pMatrix[id], pMatrix,
                    i0_biovar, len_biovar[id], biovar,
                    i0_tlag, len_tlag[id], tlag);
              PKRec<scalar> init(nCmt); init.setZero();
              PKRec<double> pred1 = VectorXd::Zero(res_d[id].cols());
              int ikeep = 0;
              for (size_t i = 0; i < em.events().size(); i++) {
                pred1 = res_d[id].row(i);
                stepper_sync(i, init, pred1, em, pred_pars..., model_pars...);
                if (em.events().keep(i)) {
                  res[id].row(ikeep) = init;
                  ikeep++;
                }
              }
            }
          }
        }
      }

      if(is_invalid) {
        throw std::runtime_error(rank_fail_msg.str());
      }
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
                     std::vector<Eigen::MatrixXd>& res,
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

      // const double invalid_res_d = -123456789987654321.0;

      using EM = EventsManager<double, double, double, double, double, double, double>;
      const int np = time.size();
      bool is_invalid = false;
      std::ostringstream rank_fail_msg;

      torsten::mpi::init();

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
      
      res.resize(np);

      for (int id = 0; id < np; ++id) {

        /* For every rank */

        res[id].resize(evid[id].size(), nCmt);
        int my_worker_id = torsten::mpi::my_worker(id, np, size);

        /* only solver rank */

        if (rank == my_worker_id) {

          EM em(nCmt, time[id], amt[id], rate[id], ii[id], evid[id], cmt[id], addl[id], ss[id], pMatrix[id], biovar[id], tlag[id]);
          auto events = em.events();
          auto model_rate = em.rates();
          auto model_amt = em.amts();
          auto model_par = em.pars();
          PKRec<double> init(nCmt); init.setZero();

          PKRec<double> pred1 = VectorXd::Zero(em.ncmt);

          try {
            for (int ik = 0; ik < em.nKeep; ik++) {
              int ibegin = ik == 0 ? 0 : em.keep_ev[ik-1] + 1;
              int iend = em.keep_ev[ik] + 1;
              for (int i = ibegin; i < iend; ++i) {
                stepper(i, init, em, pred_pars..., model_pars...);
              }
              res[id].row(ik) = init;
            }
          } catch (const std::exception& e) {
            is_invalid = true;
            res[id].setConstant(invalid_res_d);
            rank_fail_msg << "Rank " << rank << " failed solve id " << id << ": " << e.what();
          }
        }
        MPI_Ibcast(res[id].data(), res[id].size(), MPI_DOUBLE, my_worker_id, comm, &req[id]);
      }

      // make sure every rank throws in case any rank fails
      int finished = 0;
      int flag = 0;
      int index;
      while (finished != np && size > 1) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          finished++;
          if(is_invalid) continue;
          int id = index;
          if (res[id].isApproxToConstant(invalid_res_d)) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " received invalid data for id " << id;
          }
        }
      }

      // std::cout << "Torsten MPI rank: " << rank << " done" << "\n";
      if(is_invalid) {
        throw std::runtime_error(rank_fail_msg.str());
      }
    }
    
    /*
     * Data-only MPI solver that takes ragged arrays as input.
     */
    template<typename... Ts>
    static void pred(int nCmt,
                     const std::vector<int>& len,
                     const std::vector<double>& time,
                     const std::vector<double>& amt,
                     const std::vector<double>& rate,
                     const std::vector<double>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<int>& len_pMatrix,
                     const std::vector<std::vector<double> >& pMatrix,
                     const std::vector<int>& len_biovar,
                     const std::vector<std::vector<double> >& biovar,
                     const std::vector<int>& len_tlag,
                     const std::vector<std::vector<double> >& tlag,
                     std::vector<Eigen::MatrixXd>& res,
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

      // const double invalid_res_d = -123456789987654321.0;

      using EM = EventsManager<double, double, double, double, double, double, double>;
      const int np = len.size();
      bool is_invalid = false;
      std::ostringstream rank_fail_msg;

      torsten::mpi::init();

      int intialized;
      MPI_Initialized(&intialized);
      stan::math::check_greater("PredWrapper::pred", "MPI_Intialized", intialized, 0);

      MPI_Comm comm;
      comm = MPI_COMM_WORLD;
      int rank, size;
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);

      MPI_Request req[np];

      res.resize(np);

      int i0 = 0, i0_pMatrix = 0, i0_biovar = 0, i0_tlag = 0;
      for (int id = 0; id < np; ++id) {

        /* For every rank */

        res[id].resize(len[id], nCmt);
        int my_worker_id = torsten::mpi::my_worker(id, np, size);

        /* only solver rank */

        if (rank == my_worker_id) {

          EM em(nCmt, i0, len[id], time, amt, rate, ii, evid, cmt, addl, ss,
                i0_pMatrix, len_pMatrix[id], pMatrix,
                i0_biovar, len_biovar[id], biovar,
                i0_tlag, len_tlag[id], tlag);
          auto events = em.events();
          auto model_rate = em.rates();
          auto model_amt = em.amts();
          auto model_par = em.pars();
          PKRec<double> init(nCmt); init.setZero();

          PKRec<double> pred1 = VectorXd::Zero(em.ncmt);

          try {
            for (int ik = 0; ik < em.nKeep; ik++) {
              int ibegin = ik == 0 ? 0 : em.keep_ev[ik-1] + 1;
              int iend = em.keep_ev[ik] + 1;
              for (int i = ibegin; i < iend; ++i) {
                stepper(i, init, em, pred_pars..., model_pars...);
              }
              res[id].row(ik) = init;
            }
          } catch (const std::exception& e) {
            is_invalid = true;
            res[id].setConstant(invalid_res_d);
            rank_fail_msg << "Rank " << rank << " failed solve id " << id << ": " << e.what();
          }
        }
        MPI_Ibcast(res[id].data(), res[id].size(), MPI_DOUBLE, my_worker_id, comm, &req[id]);

        i0         += len[id];
        i0_pMatrix += len_pMatrix[id];
        i0_biovar  += len_biovar[id];
        i0_tlag    += len_tlag[id];
      }

      // make sure every rank throws in case any rank fails
      int finished = 0;
      int flag = 0;
      int index;
      while (finished != np && size > 1) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          finished++;
          if(is_invalid) continue;
          int id = index;
          if (res[id].isApproxToConstant(invalid_res_d)) {
            is_invalid = true;
            rank_fail_msg << "Rank " << rank << " received invalid data for id " << id;
          }
        }
      }

      // std::cout << "Torsten MPI rank: " << rank << " done" << "\n";
      if(is_invalid) {
        throw std::runtime_error(rank_fail_msg.str());
      }
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

      static bool has_warning = false;
      if (!has_warning) {
        std::cout << "Torsten Population PK solver " << "running sequentially" << "\n";
        has_warning = true;
      }

      for (int i = 0; i < np; ++i) {
        EM em(nCmt, time[i], amt[i], rate[i], ii[i], evid[i], cmt[i], addl[i], ss[i], pMatrix[i], biovar[i], tlag[i]);
        res[i].resize(em.nKeep, em.ncmt);
        pred(em, res[i], pred_pars..., model_pars...);
      }
    }

    /*
     * For population input in the form of ragged arrays,
     * addional information of the size of each individual
     * is required to locate the data in a single array for population.
     */
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename... Ts>
    static void pred(int nCmt,
                     const std::vector<int>& len,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<int>& len_pMatrix,
                     const std::vector<std::vector<T4> >& pMatrix,
                     const std::vector<int>& len_biovar,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<int>& len_tlag,
                     const std::vector<std::vector<T6> >& tlag,
                     std::vector<Eigen::Matrix<typename EventsManager<T0, T1, T2, T3, T4, T5, T6>::T_scalar, -1, -1>>& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;
      const int np = len.size();
      
      res.resize(np);

      static bool has_warning = false;
      if (!has_warning) {
        std::cout << "Torsten Population PK solver " << "running sequentially" << "\n";
        has_warning = true;
      }

      int i0 = 0, i0_pMatrix = 0, i0_biovar = 0, i0_tlag = 0;
      for (int i = 0; i < np; ++i) {
        EM em(nCmt, i0, len[i], time, amt, rate, ii, evid, cmt, addl, ss,
              i0_pMatrix, len_pMatrix[i], pMatrix,
              i0_biovar, len_biovar[i], biovar,
              i0_tlag, len_tlag[i], tlag);
        res[i].resize(em.nKeep, em.ncmt);
        pred(em, res[i], pred_pars..., model_pars...);
        i0         += len[i];
        i0_pMatrix += len_pMatrix[i];
        i0_biovar  += len_biovar[i];
        i0_tlag    += len_tlag[i];
      }
    }
#endif
  };

  template<typename T_model, typename... T_pred>
  constexpr double PredWrapper<T_model, T_pred...>::invalid_res_d;

}
#endif
