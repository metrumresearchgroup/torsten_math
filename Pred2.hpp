#ifndef STAN_MATH_TORSTEN_PKMODEL_REFACTOR_PRED_HPP
#define STAN_MATH_TORSTEN_PKMODEL_REFACTOR_PRED_HPP

#include <Eigen/Dense>
#include <vector>
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
    static void pred(EventsManager<T_em...>& em,
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

      PKRec<scalar> zeros = PKRec<scalar>::Zero(res.cols());
      PKRec<scalar> init = zeros;
      auto dt = events.time(0);
      auto tprev = events.time(0);
      Matrix<scalar, Dynamic, 1> pred1;
      int ikeep = 0;

      for (size_t i = 0; i < events.size(); i++) {
        if (events.is_reset(i)) {
          dt = 0;
          init = zeros;
        } else {
          dt = events.time(i) - tprev;
          decltype(tprev) model_time = tprev;

          // FIX ME: we need a better way to relate model type to parameter type
          T_model pkmodel {model_time, init, model_rate[i], model_par[i], model_pars...};

          pred1 = pkmodel.solve(dt, pred_pars...);
          init = pred1;
        }

        if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
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
    template<typename T_E, typename... Ts,
             typename std::enable_if_t<
               stan::is_var<stan::return_type<typename T_E::T_amt, typename T_E::T_rate, typename T_E::T_par>::type>::value>* = nullptr> //NOLINT
    static void pred(std::vector<T_E>& em,
                     std::vector<Eigen::Matrix<typename EventsManager<T_em...>::T_scalar, -1, -1>>& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;

      static const char* caller = "PredWrapper::pred";
      stan::math::check_less(caller, "population size", em.size(), res.size());

      // make sure MPI is on
      int intialized;
      MPI_Initialized(&intialized);
      stan::math::check_greater("pk_integrate_ode_bdf", "MPI_Intialized", intialized, 0);

      MPI_Comm comm;
      comm = MPI_COMM_WORLD;
      int rank, size;
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);

      size_t np = em.size();
      MPI_Request req[np];

      std::vector<Matrix<double, Dynamic, Dynamic> > res_mpi(em.size());
      for (int i = 0; i < np; ++i) {
        res[i].resize(em[i].nKeep, em[i].ncmt);
        int nsys = em[i].ncmt * (em[i].vars(0).size() + 1);
        res_mpi[i].resize(em[i].nKeep, nsys);
        int my_worker_id = torsten::mpi::my_worker(i, np, size);      
        if(rank == my_worker_id) {
          pred(em[i], res[i], pred_pars..., model_pars...);
          try {
            stan::math::start_nested();
            std::vector<double> g;
            for (size_t j = 0; j < em[i].ncmt; ++j) {
              for (size_t k = 0; k < em[i].nKeep; ++k) {
                stan::math::set_zero_all_adjoints_nested();
                res_mpi[i](k, j * nsys) = res[i](k, j).val();
                res[i](k, j).grad(em[i].vars(k), g);
                for (size_t l = 0; l < g.size(); ++l) {
                  res_mpi[i](k, j * nsys + l + 1) = g[l];
                }
              }
            }
          } catch (const std::exception& e) {
            std::an::math::recover_memory_nested();
            throw;
          }
          stan::math::recover_memory_nested();

          MPI_Ibcast(res_mpi[i].data(), res_mpi[i].size(), MPI_DOUBLE, my_worker_id, comm, &req[i]);
        }
      }

      int finished = 0;
      int flag = 0;
      int index;
      while(finished != np) {
        MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
          int i = index;
          int nsys = em[i].ncmt * (em[i].vars(0).size() + 1);
          std::vector<double> g(nsys - 1);
          int my_worker_id = torsten::mpi::my_worker(i, np, size);
          if(rank != my_worker_id) {
            for (size_t j = 0; j < em[i].ncmt; ++j) {
              for (size_t k = 0; k < em[i].nKeep; ++k) {
                for (int l = 0 ; l < g.size(); ++l) g[l] = res_mpi[i](k, j * nsys + l + 1);
                res[i](k, j) = precomputed_gradients(res_i[i](k, j * nsys), em[i].vars(k), g);
              }
            }
          }
          finished++;
        }
      }
    }

    template<typename T_E, typename... Ts,
             typename std::enable_if_t<
               !stan::is_var<stan::return_type<typename T_E::T_amt, typename T_E::T_rate, typename T_E::T_par>::type>::value>* = nullptr> //NOLINT
    static void pred(std::vector<T_E>& em,
                     std::vector<Eigen::Matrix<double, -1, -1> >& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      static const char* caller = "PredWrapper::pred";
      stan::math::check_less(caller, "population size", em.size(), res.size());

      // make sure MPI is on
      int intialized;
      MPI_Initialized(&intialized);
      stan::math::check_greater("pk_integrate_ode_bdf", "MPI_Intialized", intialized, 0);

      MPI_Comm comm;
      comm = MPI_COMM_WORLD;
      int rank, size;
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);

      size_t np = em.size();
      MPI_Request req[np];

      for (int i = 0; i < np; ++i) {
        res[i].resize(em[i].nKeep, em[i].ncmt);
        int my_worker_id = torsten::mpi::my_worker(i, np, size);      
        if(rank == my_worker_id) {
          pred(em[i], res[i], pred_pars..., model_pars...);
        }
        MPI_Ibcast(res[i].data(), res[i].size(), MPI_DOUBLE, my_worker_id, comm, &req[i]);

        int finished = 0;
        int flag = 0;
        int index;
        while(finished != np) {
          MPI_Testany(np, req, &index, &flag, MPI_STATUS_IGNORE);
          if(flag) {
            finished++;
          }
        }
      }
    }
#else
    template<typename... T_em, typename... Ts>
    static void pred(std::vector<EventsManager<T_em...> >& em,
                     std::vector<Eigen::Matrix<typename EventsManager<T_em...>::T_scalar, -1, -1> >& res,
                     const T_pred... pred_pars,
                     const Ts... model_pars) {
      static const char* caller = "PredWrapper::pred";
      stan::math::check_less_or_equal(caller, "population size", em.size(), res.size());

      for (size_t i = 0; i < em.size(); ++i) {
        res[i].resize(em[i].nKeep, em[i].ncmt);
        pred(em[i], res[i], pred_pars..., model_pars...);
      }
    }    
#endif    
  };
}

#endif
