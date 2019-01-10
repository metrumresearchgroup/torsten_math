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
  template<typename T_model>
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
     * @tparam T_time type of scalar for time
     * @tparam T_amt type of scalar for amount
     * @tparam T_rate type of scalar for rate
     * @tparam T_ii type of scalar for interdose interval
     * @tparam T_parameters type of scalar for the ODE parameters
     * @tparam T_biovar type of scalar for bio-variability parameters
     * @tparam T_tlag type of scalar for lag times parameters
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
    template<typename T_eh, typename T_ph, typename T_rate, typename T_amt,typename scalar,
             typename T_parameters,
             typename F_one,
             typename F_SS,
             PkOdeIntegratorId It,
             typename... Ts>
    void Pred2(const T_eh& events, const T_ph& parameters,
          const std::vector<std::vector<T_rate> >& model_rate,
          const std::vector<T_amt>& model_amt,
          Eigen::Matrix<scalar, -1, -1>& pred,
          const int& nCmt,
          const std::vector<Eigen::Matrix<T_parameters, Eigen::Dynamic, Eigen::Dynamic> >& system,
          const F_one& Pred1,
          const F_SS& PredSS,
          const PkOdeIntegrator<It>& integrator,
          const Ts... pars) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using boost::math::tools::promote_args;
      using std::vector;
      using::stan::math::multiply;
      using refactor::PKRec;

      PKRec<scalar> zeros = PKRec<scalar>::Zero(nCmt);
      PKRec<scalar> init = zeros;
      auto dt = events.time(0);
      auto tprev = events.time(0);
      Matrix<scalar, Dynamic, 1> pred1;
      int ikeep = 0;

      for (int i = 0; i < events.size(); i++) {
        auto parameter = parameters.GetModelParameters(i);
        if (events.is_reset(i)) {
          dt = 0;
          init = zeros;
        } else {
          dt = events.time(i) - tprev;
          using model_type = T_model;
          decltype(tprev) model_time = tprev;

          // std::vector<T_parameters> model_par = parameter.get_RealParameters();

          // FIX ME: we need a better way to relate model type to parameter type
          std::vector<T_parameters> model_par = model_type::get_param(parameter);
          model_type pkmodel {model_time, init, model_rate[i], model_par, pars...};

          pred1 = pkmodel.solve(dt, integrator);
          // pred1 = Pred1(dt, parameter, init, rate2.get_rate());
          init = pred1;
        }

        if ((events.is_dosing(i) && (events.ss(i) == 1 || events.ss(i) == 2)) || events.ss(i) == 3) {  // steady state event
          using model_type = T_model;
          decltype(tprev) model_time = events.time(i); // FIXME: time is not t0 but for adjust within SS solver
          // auto model_par = parameter.get_RealParameters();
          // FIX ME: we need a better way to relate model type to parameter type
          std::vector<T_parameters> model_par = model_type::get_param(parameter);
          model_type pkmodel {model_time, init, model_rate[i], model_par, pars...};
          pred1 = multiply(pkmodel.solve(model_amt[i], //NOLINT
                                         events.rate(i),
                                         events.ii(i),
                                         events.cmt(i),
                                         integrator),
                           scalar(1.0));
          // pred1 = multiply(PredSS(parameter,
          //                         parameters.GetValueBio(i, events.cmt(i) - 1)
          //                           * events.amt(i),
          //                         events.rate(), events.ii(),
          //                         events.cmt(i)),
          //                  scalar(1.0));


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
