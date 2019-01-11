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
    void pred(EventsManager<T_em...>& em,
              Eigen::Matrix<typename EventsManager<T_em...>::T_scalar, -1, -1>& pred,
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

      PKRec<scalar> zeros = PKRec<scalar>::Zero(pred.cols());
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
          pred.row(ikeep) = init;
          ikeep++;
        }
        tprev = events.time(i);
      }
    }
  };
}

#endif
