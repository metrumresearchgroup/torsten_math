#ifndef STAN_MATH_TORSTEN_PKMODEL_PRED_PREDSS_LINCPT_HPP
#define STAN_MATH_TORSTEN_PKMODEL_PRED_PREDSS_LINCPT_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/math/rev/mat.hpp>
#include <stan/math/fwd/mat.hpp>
#include <iostream>

using std::vector;
using boost::math::tools::promote_args;
using Eigen::Matrix;
using Eigen::Dynamic;

/**
 * General compartment model using built-in ODE solver. 
 * Calculate amount in each compartment at the end of a
 * steady-state dosing interval or during a steady-state
 * constant input (if ii=0)
 * 
 * 
 * Model using numerical ODE solver are not available for
 * steady state. If data contains a steady state event, 
 * abort.
 * DEV - use invalid / error message 
 * 
 * @tparam T_time type of scalar for time
 * @tparam T_amt type of scalar for amount
 * @tparam T_rate type of scalar for rate
 * @tparam T_ii type of scalar for interdose interval
 * @tparam T_system type of elements of Matrix that describes
 * ODE 
 * @param[in] parameter model parameters at current event
 * @param[in] rate
 * @param[in] ii interdose interval
 * @param[in] cmt compartment in which the event occurs 
 * @param[in] f functor for base ordinary differential equation that defines 
 *   compartment model 
 * @return an eigen vector that contains predicted amount in each compartment 
 *   at the current event. 
 */
template<typename T_amt, typename T_rate, typename T_ii, typename T_system>
Matrix<typename promote_args<T_amt, T_rate, T_ii, T_system>::type, 1, Dynamic>
PredSS_linCpt(const T_amt& amt, 
		   	  const T_rate& rate,
		   	  const T_ii& ii, 
		   	  const int& cmt,
		   	  const Eigen::Matrix<T_system, Eigen::Dynamic,
		   	    Eigen::Dynamic> system) {

  using stan::math::matrix_exp;
  using stan::math::mdivide_left;

  typedef typename promote_args<T_amt, T_rate, T_ii, T_system>::type scalar; 

  int nCmt = system.rows();
  Matrix<scalar, Dynamic, Dynamic> workMatrix, ii_system = ii * system; // Check
  Matrix<scalar, 1, Dynamic> pred(nCmt);
  pred.setZero();
  Matrix<scalar, Dynamic, 1> amounts(nCmt);
  amounts.setZero();

  if(rate == 0) {  // bolus dose
    amounts(cmt - 1) = amt;    
    workMatrix = - matrix_exp(ii_system);
    for(int i = 0; i < nCmt; i++) workMatrix(i, i) += 1;
    amounts = mdivide_left(workMatrix, amounts); // FIXME - check singularity    
    pred = matrix_exp(ii_system) * amounts;    
  }

  else if (ii > 0) {  // multiple truncated infusions
    amounts(cmt - 1) = rate;
    double t = amt / rate;
    assert(t <= ii);
    amounts = mdivide_left(system, amounts);
    Matrix<scalar, Dynamic, Dynamic> t_system = t * system; // Check - case where t and system don't have the same type
    pred = matrix_exp(t_system) * amounts;
    pred -= amounts;

    workMatrix = - matrix_exp(ii_system);
    for(int i = 0; i < nCmt; i++) workMatrix(i, i) += 1;
    pred = mdivide_left(workMatrix, pred);
    t = t - ii;
    t_system = t * system;
    pred = matrix_exp(t_system) * pred;      
  }

  else {  // constant infusion
    amounts(cmt - 1) -= rate;
    pred = mdivide_left(system, amounts);
  }

  return pred;
}

#endif
