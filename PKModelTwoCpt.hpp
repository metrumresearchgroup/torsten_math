#ifndef STAN_MATH_TORSTEN_PKMODELTWOCPT_HPP
#define STAN_MATH_TORSTEN_PKMODELTWOCPT_HPP

#include <Eigen/Dense>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/torsten/PKModel/PKModel.hpp>
#include <stan/math/torsten/PKModel/dummy_ode.hpp>
#include <string>
#include <vector>

/**
 * Computes the predicted amounts in each compartment at each event
 * for a two compartment model with first oder absorption.
 *
 * @tparam T0 type of scalars for the model parameters.
 * @tparam T1 type of scalar for time of events.
 * @tparam T2 type of scalar for amount at each event.
 * @tparam T3 type of scalar for rate at each event.
 * @tparam T4 type of scalar for inter-dose inteveral at each event.
 * @param[in] pMatrix parameters at each event
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
 * @return a matrix with predicted amount in each compartment
 *         at each event.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  T4>::type, Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector< Eigen::Matrix<T0, Eigen::Dynamic, 1> >&
                pMatrix,
              const std::vector<T1>& time,
              const std::vector<T2>& amt,
              const std::vector<T3>& rate,
              const std::vector<T4>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss) {
  using std::vector;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using boost::math::tools::promote_args;

  PKModel model("TwoCptModel");  // Define class of model
  static const char* function("PKModelTwoCpt");

  // Check arguments
  pmetricsCheck(pMatrix, time, amt, rate, ii, evid, cmt, addl, ss, function,
    model);
  for (int i = 0; i < pMatrix.size(); i++) {
    stan::math::check_positive_finite(function, "PK parameter CL",
      pMatrix[i](0, 0));
    stan::math::check_positive_finite(function, "PK parameter Q",
      pMatrix[i](1, 0));
    stan::math::check_positive_finite(function, "PK parameter V2",
      pMatrix[i](2, 0));
    stan::math::check_positive_finite(function, "PK parameter V3",
      pMatrix[i](3, 0));
    stan::math::check_positive_finite(function, "PK parameter ka",
      pMatrix[i](4, 0));
  }
  std::string message4 = ", but must equal the number of parameters in the model: " // NOLINT
    + boost::lexical_cast<std::string>(model.GetNParameter()) + "!";
  const char* length_error4 = message4.c_str();
  if (!(pMatrix[0].size() == model.GetNParameter()))
    stan::math::invalid_argument(function,
    "The number of parameters per event (length of a vector in the first argument) is", // NOLINT
    pMatrix[0].size(), "", length_error4);

  // Construct Pred functions for the model.
  Pred1_structure new_Pred1("TwoCptModel");
  PredSS_structure new_PredSS("TwoCptModel");
  Pred1 = new_Pred1;
  PredSS = new_PredSS;

  // Construct dummy matrix for last argument of pred
  Matrix<double, Dynamic, Dynamic> dummy_system;
  vector<Matrix<double, Dynamic, Dynamic> >
    dummy_systems(1, dummy_system);

  Matrix <typename promote_args<typename promote_args<T0, T1, T2, T3,
    T4>::type, double>::type, Dynamic, Dynamic> pred;
  pred = Pred(pMatrix, time, amt, rate, ii, evid, cmt, addl, ss, model,
    dummy_ode(), dummy_systems);

  return pred;
}

#endif
