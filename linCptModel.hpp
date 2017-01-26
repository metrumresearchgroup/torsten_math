#ifndef STAN_MATH_TORSTEN_LINCPTMODEL_HPP
#define STAN_MATH_TORSTEN_LINCPTMODEL_HPP

#include <Eigen/Dense>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/torsten/PKModel/PKModel.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <vector>

/**
 * Computes the predicted amounts in each compartment at each event
 * for a compartment model, described by a linear system of ordinary
 * differential equations. Uses the stan::math::matrix_exp 
 * function.
 *
 * <b>Warning:</b> This prototype does not handle steady state events. 
 *
 * @tparam T0 type of scalar for time of events. 
 * @tparam T1 type of scalar for amount at each event.
 * @tparam T2 type of scalar for rate at each event.
 * @tparam T3 type of scalar for inter-dose inteveral at each event.
 * @tparam T4 type of scalar for matrix describing linear ODE system.
 * @tparam T5 type of scalars for bio-variability parameters.
 * @tparam T6 type of scalars for tlag parameters 
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
 * @param[in] cmt compartment number at each event 
 * @param[in] addl additional dosing at each event 
 * @param[in] ss steady state approximation at each event (0: no, 1: yes)
 * between time-points
 * @param[in] system square matrix describing the linear system of ODEs
 * @param[in] bio-variability at each event
 * @param[in] lag times at each event
 * @return a matrix with predicted amount in each compartment 
 * at each event.
 */
template <typename T0, typename T1, typename T2, typename T3,
  typename T4, typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
linCptModel(const std::vector<T0>& time,
            const std::vector<T1>& amt,
            const std::vector<T2>& rate,
            const std::vector<T3>& ii,
            const std::vector<int>& evid,
            const std::vector<int>& cmt,
            const std::vector<int>& addl,
            const std::vector<int>& ss,
            const std::vector< Eigen::Matrix<T4, Eigen::Dynamic,
              Eigen::Dynamic> >& system,
            const std::vector<std::vector<T5> >& biovar,
            const std::vector<std::vector<T6> >& tlag) {
  using std::vector;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using boost::math::tools::promote_args;

  static const char* function("linCptModel");
  for (size_t i = 0; i < system.size(); i++)
    stan::math::check_square(function, "system matrix", system[i]);
  int nCmt = system[0].cols();

  // int nParameters = pMatrix[0].size();
  PKModel model(0, nCmt);  // CHECK - what should I use for nParameters?

  // Check arguments
  std::vector<double> parameters_dummy(0);
  std::vector<std::vector<double> > pMatrix_dummy(1, parameters_dummy);
  pmetricsCheck(time, amt, rate, ii, evid, cmt, addl, ss,
                pMatrix_dummy, biovar, tlag, function, model);

  // define functors used in Pred()
  Pred1_structure new_Pred1("linCptModel");
  PredSS_structure new_PredSS("linCptModel");
  Pred1 = new_Pred1;
  PredSS = new_PredSS;

  Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
    typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
    Dynamic, Dynamic> pred;
   pred = Pred(time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_dummy,
               biovar, tlag, model, dummy_ode(), system);

  return pred;
}

/*
 * Overload function to allow user to pass an std::vector for 
 * pMatrix.
 */ /*
template <typename T0, typename T1, typename T2, typename T3,
  typename T4, typename T5>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  T4>::type, Eigen::Dynamic, Eigen::Dynamic>
linCptModel(const std::vector< Eigen::Matrix<T0, Eigen::Dynamic,
              Eigen::Dynamic> >& system,
            const std::vector<T1>& pMatrix,
            const std::vector<T2>& time,
            const std::vector<T3>& amt,
            const std::vector<T4>& rate,
            const std::vector<T5>& ii,
            const std::vector<int>& evid,
            const std::vector<int>& cmt,
            const std::vector<int>& addl,
            const std::vector<int>& ss) {
  std::vector<std::vector<T1> > vec_pMatrix(1);
  vec_pMatrix[0] = pMatrix;

  return linCptModel(system,
    vec_pMatrix, time, amt, rate, ii, evid, cmt, addl, ss);
}

template <typename T0, typename T1, typename T2, typename T3,
  typename T4, typename T5>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  T4>::type, Eigen::Dynamic, Eigen::Dynamic>
linCptModel(const Eigen::Matrix<T0, Eigen::Dynamic,
              Eigen::Dynamic>& system,
            const std::vector<std::vector<T1> >& pMatrix,
            const std::vector<T2>& time,
            const std::vector<T3>& amt,
            const std::vector<T4>& rate,
            const std::vector<T5>& ii,
            const std::vector<int>& evid,
            const std::vector<int>& cmt,
            const std::vector<int>& addl,
            const std::vector<int>& ss) {
  std::vector<Eigen::Matrix<T0, Eigen::Dynamic,
              Eigen::Dynamic> > vec_system(1);
  vec_system[0] = system;

  return linCptModel(vec_system,
    pMatrix, time, amt, rate, ii, evid, cmt, addl, ss);
}

template <typename T0, typename T1, typename T2, typename T3,
  typename T4, typename T5>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  T4>::type, Eigen::Dynamic, Eigen::Dynamic>
linCptModel(const Eigen::Matrix<T0, Eigen::Dynamic,
              Eigen::Dynamic>& system,
            const std::vector<T1>& pMatrix,
            const std::vector<T2>& time,
            const std::vector<T3>& amt,
            const std::vector<T4>& rate,
            const std::vector<T5>& ii,
            const std::vector<int>& evid,
            const std::vector<int>& cmt,
            const std::vector<int>& addl,
            const std::vector<int>& ss) {
  std::vector<Eigen::Matrix<T0, Eigen::Dynamic,
              Eigen::Dynamic> > vec_system(1);
  vec_system[0] = system;

  std::vector<std::vector<T1> > vec_pMatrix(1);
  vec_pMatrix[0] = pMatrix;

  return linCptModel(vec_system,
    vec_pMatrix, time, amt, rate, ii, evid, cmt, addl, ss);
  } 
*/

#endif
