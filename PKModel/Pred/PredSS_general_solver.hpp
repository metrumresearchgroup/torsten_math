#ifndef STAN_MATH_TORSTEN_PKMODEL_PRED_PREDSS_GENERAL_SOLVER_HPP
#define STAN_MATH_TORSTEN_PKMODEL_PRED_PREDSS_GENERAL_SOLVER_HPP

#include <iostream>

/**
 * General compartment model using built-in ODE solver.
 * Calculate amount in each compartment at the end of a steady-state dosing interval
 * or during a steady-state constant input (if ii=0)
 *
 *
 * Model using numerical ODE solver are not available for steady state. If data contains a
 * steady state event, abort. DEV - use invalid / error message
 *
 *	 @tparam T_time type of scalar for time
 *	 @tparam T_amt type of scalar for amount
 *	 @tparam T_rate type of scalar for rate
 *	 @tparam T_ii type of scalar for interdose interval
 *	 @tparam T_parameters type of scalar for model parameters
 *	 @tparam F type of ODE system function
 *	 @param[in] parameter model parameters at current event
 *	 @param[in] rate
 *	 @param[in] ii interdose interval
 *	 @param[in] cmt compartment in which the event occurs
 *	 @param[in] f functor for base ordinary differential equation that defines
 *              compartment model
 *   @return an eigen vector that contains predicted amount in each compartment
 *           at the current event.
 */
template<typename T_time, typename T_amt, typename T_rate,
  typename T_ii, typename T_parameters, typename T_system, typename F>
Eigen::Matrix<typename boost::math::tools::promote_args<T_time, T_amt, T_rate,
  typename boost::math::tools::promote_args< T_ii, T_parameters>::type>::type,
  1, Eigen::Dynamic>
PredSS_general_solver(const ModelParameters<T_time, T_parameters, T_system>&
                        parameter,
                      const T_amt& amt,
                      const T_rate& rate,
                      const T_ii& ii,
                      const int& cmt,
                      const F& f) {
  typedef typename boost::math::tools::promote_args< T_time, T_amt, T_rate,
    typename boost::math::tools::promote_args< T_ii, T_parameters>::type>::
    type scalar;

  std::cout << "ERROR: General Compartment Model using ODE integrator"
            << " (bdf or rk45) does not handle Steady State events."
            << std::endl;
  abort();  // FIX ME: throw an exception, not abort.

  Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred
    = Eigen::Matrix<scalar, 1, Eigen::Dynamic>::Zero(3);
  return pred;
}

#endif
