#ifndef STAN_MATH_TORSTEN_PKMODELTWOCPT2_HPP
#define STAN_MATH_TORSTEN_PKMODELTWOCPT2_HPP

#include <Eigen/Dense>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/torsten/Pred2.hpp>
#include <stan/math/torsten/events_manager.hpp>
#include <stan/math/torsten/PKModel/PKModel.hpp>
#include <stan/math/torsten/PKModel/Pred/Pred1_twoCpt.hpp>
#include <stan/math/torsten/PKModel/Pred/PredSS_twoCpt.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
// #include <stan/math/torsten/pk_twocpt_solver.hpp>
// #include <stan/math/torsten/pk_twocpt_solver_ss.hpp>
#include <string>
#include <vector>

namespace torsten {

/**
 * Computes the predicted amounts in each compartment at each event
 * for a two compartment model with first oder absorption.
 *
 * @tparam T0 type of scalar for time of events.
 * @tparam T1 type of scalar for amount at each event.
 * @tparam T2 type of scalar for rate at each event.
 * @tparam T3 type of scalar for inter-dose inteveral at each event.
 * @tparam T4 type of scalars for the model parameters.
 * @tparam T5 type of scalars for the 
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
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<std::vector<T4> >& pMatrix,
              const std::vector<std::vector<T5> >& biovar,
              const std::vector<std::vector<T6> >& tlag) {
  using std::vector;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using boost::math::tools::promote_args;
  using stan::math::check_positive_finite;
  using refactor::PKRec;

  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  int nParms = refactor::PKTwoCptModel<double, double, double, double>::Npar;
  static const char* function("PKModelTwoCpt");

  // Check arguments
  torsten::pmetricsCheck(time, amt, rate, ii, evid, cmt, addl, ss,
                pMatrix, biovar, tlag, function);
  for (size_t i = 0; i < pMatrix.size(); i++) {
    check_positive_finite(function, "PK parameter CL", pMatrix[i][0]);
    check_positive_finite(function, "PK parameter Q", pMatrix[i][1]);
    check_positive_finite(function, "PK parameter V2", pMatrix[i][2]);
    check_positive_finite(function, "PK parameter V3", pMatrix[i][3]);
  }
  std::string message4 = ", but must equal the number of parameters in the model: " // NOLINT
    + boost::lexical_cast<std::string>(nParms) + "!";
  const char* length_error4 = message4.c_str();
  if (!(pMatrix[0].size() == (size_t) nParms))
    stan::math::invalid_argument(function,
    "The number of parameters per event (length of a vector in the first argument) is", // NOLINT
    pMatrix[0].size(), "", length_error4);

  // FIX ME - we want to check every array of pMatrix, not
  // just the first one (at index 0)
  std::string message5 = ", but must equal the number of parameters in the model: " // NOLINT
  + boost::lexical_cast<std::string>(nParms) + "!";
  const char* length_error5 = message5.c_str();
  if (!(pMatrix[0].size() == (size_t) nParms))
    stan::math::invalid_argument(function,
    "The number of parameters per event (length of a vector in the ninth argument) is", // NOLINT
    pMatrix[0].size(), "", length_error5);

  std::string message6 = ", but must equal the number of compartments in the model: " // NOLINT
  + boost::lexical_cast<std::string>(nCmt) + "!";
  const char* length_error6 = message6.c_str();
  if (!(biovar[0].size() == (size_t) nCmt))
    stan::math::invalid_argument(function,
    "The number of biovariability parameters per event (length of a vector in the tenth argument) is", // NOLINT
    biovar[0].size(), "", length_error6);

  if (!(tlag[0].size() == (size_t) nCmt))
    stan::math::invalid_argument(function,
    "The number of lag times parameters per event (length of a vector in the eleventh argument) is", // NOLINT
    tlag[0].size(), "", length_error5);

  // Construct dummy matrix for last argument of pred
  Matrix<T4, Dynamic, Dynamic> dummy_system;
  vector<Matrix<T4, Dynamic, Dynamic> >
    dummy_systems(1, dummy_system);

#ifdef OLD_TORSTEN
  return Pred(time, amt, rate, ii, evid, cmt, addl, ss,
              pMatrix, biovar, tlag,
              nCmt, dummy_systems,
              Pred1_twoCpt(), PredSS_twoCpt());
#else
  using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;
  EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);

  Matrix<typename EM::T_scalar, Dynamic, Dynamic> pred =
    Matrix<typename EM::T_scalar, Dynamic, Dynamic>::Zero(em.nKeep, em.ncmt);

  using model_type = refactor::PKTwoCptModel<typename EM::T_time, typename EM::T_scalar, typename EM::T_rate, typename EM::T_par>;
  PredWrapper<model_type> pr;
  pr.pred(em, pred);
  return pred;

#endif
}

/**
 * Overload function to allow user to pass an std::vector for pMatrix.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<T4>& pMatrix,
              const std::vector<std::vector<T5> >& biovar,
              const std::vector<std::vector<T6> >& tlag) {
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       vec_pMatrix, biovar, tlag);
}

/**
* Overload function to allow user to pass an std::vector for pMatrix,
* and biovar.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<T4>& pMatrix,
              const std::vector<T5>& biovar,
              const std::vector<std::vector<T6> >& tlag) {
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);
  std::vector<std::vector<T5> > vec_biovar(1, biovar);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       vec_pMatrix, vec_biovar, tlag);
}

/**
* Overload function to allow user to pass an std::vector for pMatrix,
* and tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<T4>& pMatrix,
              const std::vector<std::vector<T5> >& biovar,
              const std::vector<T6>& tlag) {
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       vec_pMatrix, biovar, vec_tlag);
}

/**
* Overload function to allow user to pass an std::vector for pMatrix,
* biovar, and tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<T4>& pMatrix,
              const std::vector<T5>& biovar,
              const std::vector<T6>& tlag) {
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);
  std::vector<std::vector<T5> > vec_biovar(1, biovar);
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       vec_pMatrix, vec_biovar, vec_tlag);
}

/**
* Overload function to allow user to pass an std::vector for biovar.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<std::vector<T4> >& pMatrix,
              const std::vector<T5>& biovar,
              const std::vector<std::vector<T6> >& tlag) {
  std::vector<std::vector<T5> > vec_biovar(1, biovar);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, vec_biovar, tlag);
}

/**
* Overload function to allow user to pass an std::vector for biovar,
* and tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<std::vector<T4> >& pMatrix,
              const std::vector<T5>& biovar,
              const std::vector<T6>& tlag) {
  std::vector<std::vector<T5> > vec_biovar(1, biovar);
  std::vector<std::vector<T6> >vec_tlag(1, tlag);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, vec_biovar, vec_tlag);
}

/**
* Overload function to allow user to pass an std::vector for tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
PKModelTwoCpt(const std::vector<T0>& time,
              const std::vector<T1>& amt,
              const std::vector<T2>& rate,
              const std::vector<T3>& ii,
              const std::vector<int>& evid,
              const std::vector<int>& cmt,
              const std::vector<int>& addl,
              const std::vector<int>& ss,
              const std::vector<std::vector<T4> >& pMatrix,
              const std::vector<std::vector<T5> >& biovar,
              const std::vector<T6>& tlag) {
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, vec_tlag);
}

  /* 
   * For population models, we follow the call signature
   * with the only change that each argument adds an additional
   * level of vector. The size of that vector is the siez of
   * the population.
   */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
std::vector<Eigen::Matrix<typename EventsManager<T0, T1, T2, T3, T4, T5, T6>::T_scalar, // NOLINT
                          Eigen::Dynamic, Eigen::Dynamic> >
popPKModelTwoCpt(const std::vector<std::vector<T0> >& time,
                 const std::vector<std::vector<T1> >& amt,
                 const std::vector<std::vector<T2> >& rate,
                 const std::vector<std::vector<T3> >& ii,
                 const std::vector<std::vector<int> >& evid,
                 const std::vector<std::vector<int> >& cmt,
                 const std::vector<std::vector<int> >& addl,
                 const std::vector<std::vector<int> >& ss,
                 const std::vector<std::vector<std::vector<T4> > >& pMatrix,
                 const std::vector<std::vector<std::vector<T5> > >& biovar,
                 const std::vector<std::vector<std::vector<T6> > >& tlag) {

  int np = time.size();
  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  static const char* caller("PKModelTwoCpt");
  stan::math::check_consistent_sizes(caller, "time", time, "amt",     amt);
  stan::math::check_consistent_sizes(caller, "time", time, "rate",    rate);
  stan::math::check_consistent_sizes(caller, "time", time, "ii",      ii);
  stan::math::check_consistent_sizes(caller, "time", time, "evid",    evid);
  stan::math::check_consistent_sizes(caller, "time", time, "cmt",     cmt);
  stan::math::check_consistent_sizes(caller, "time", time, "addl",    addl);
  stan::math::check_consistent_sizes(caller, "time", time, "ss",      ss);
  stan::math::check_consistent_sizes(caller, "time", time, "pMatrix", pMatrix);
  stan::math::check_consistent_sizes(caller, "time", time, "biovar",  biovar);
  stan::math::check_consistent_sizes(caller, "time", time, "tlag",    tlag);

  using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;

  using model_type = refactor::PKTwoCptModel<typename EM::T_time, typename EM::T_scalar, typename EM::T_rate, typename EM::T_par>;
  PredWrapper<model_type> pr;

  std::vector<Eigen::Matrix<typename EM::T_scalar, -1, -1>> pred(np);

  pr.pred(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag, pred);

  return pred;
}

  /* 
   * For population models, we follow the call signature
   * but add the arrays of the length of each individual's data. 
   * The size of that vector is the siez of
   * the population.
   */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
std::vector<Eigen::Matrix<typename EventsManager<T0, T1, T2, T3, T4, T5, T6>::T_scalar, // NOLINT
                          Eigen::Dynamic, Eigen::Dynamic> >
popPKModelTwoCpt(const std::vector<int>& len,
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
                 const std::vector<std::vector<T6> >& tlag) {
  using stan::math::check_consistent_sizes;
  using stan::math::check_greater_or_equal;

  int np = len.size();
  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  static const char* caller("PKModelTwoCpt");
  check_consistent_sizes(caller, "time", time, "amt",     amt);
  check_consistent_sizes(caller, "time", time, "rate",    rate);
  check_consistent_sizes(caller, "time", time, "ii",      ii);
  check_consistent_sizes(caller, "time", time, "evid",    evid);
  check_consistent_sizes(caller, "time", time, "cmt",     cmt);
  check_consistent_sizes(caller, "time", time, "addl",    addl);
  check_consistent_sizes(caller, "time", time, "ss",      ss);
  check_consistent_sizes(caller, "population", len, "parameters", len_pMatrix);
  check_consistent_sizes(caller, "population", len, "biovar", len_biovar);
  check_consistent_sizes(caller, "population", len, "tlag", len_tlag);

  size_t s;
  s = 0; for (auto& i : len)         {s += i;} check_greater_or_equal(caller, "time size", time.size(), s);
  s = 0; for (auto& i : len_pMatrix) {s += i;} check_greater_or_equal(caller, "pMatrix size", pMatrix.size(), s);
  s = 0; for (auto& i : len_biovar)  {s += i;} check_greater_or_equal(caller, "biovar size", biovar.size(), s);
  s = 0; for (auto& i : len_tlag)    {s += i;} check_greater_or_equal(caller, "tlag size", tlag.size(), s);

  using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;

  using model_type = refactor::PKTwoCptModel<typename EM::T_time, typename EM::T_scalar, typename EM::T_rate, typename EM::T_par>;
  PredWrapper<model_type> pr;

  std::vector<Eigen::Matrix<typename EM::T_scalar, -1, -1>> pred(np);

  pr.pred(nCmt, len, time, amt, rate, ii, evid, cmt, addl, ss,
          len_pMatrix, pMatrix,
          len_biovar, biovar,
          len_tlag, tlag,
          pred);

  return pred;
}

}
#endif
