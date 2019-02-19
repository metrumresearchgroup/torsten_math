#ifndef STAN_MATH_TORSTEN_REFACTOR_GENERALODEMODEL_BDF_HPP
#define STAN_MATH_TORSTEN_REFACTOR_GENERALODEMODEL_BDF_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/events_manager.hpp>
#include <stan/math/torsten/PKModel/functors/general_functor.hpp>
#include <stan/math/torsten/PKModel/PKModel.hpp>
#include <stan/math/torsten/PKModel/Pred/Pred1_general.hpp>
#include <stan/math/torsten/PKModel/Pred/PredSS_general.hpp>
#include <stan/math/torsten/pk_ode_model.hpp>
#include <stan/math/torsten/Pred2.hpp>
#include <boost/math/tools/promotion.hpp>
#include <vector>

namespace torsten {

/**
 * Computes the predicted amounts in each compartment at each event
 * for a general compartment model, defined by a system of ordinary
 * differential equations. Uses the stan::math::integrate_ode_bdf 
 * function. 
 *
 * <b>Warning:</b> This prototype does not handle steady state events. 
 *
 * @tparam T0 type of scalar for time of events. 
 * @tparam T1 type of scalar for amount at each event.
 * @tparam T2 type of scalar for rate at each event.
 * @tparam T3 type of scalar for inter-dose inteveral at each event.
 * @tparam T4 type of scalars for the model parameters.
 * @tparam T5 type of scalars for the bio-variability parameters.
 * @tparam T6 type of scalars for the model tlag parameters.
 * @tparam F type of ODE system function.
 * @param[in] f functor for base ordinary differential equation that defines 
 *            compartment model.
 * @param[in] nCmt number of compartments in model
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
 * @param[in] rel_tol relative tolerance for the Boost ode solver 
 * @param[in] abs_tol absolute tolerance for the Boost ode solver
 * @param[in] max_num_steps maximal number of steps to take within 
 *            the Boost ode solver 
 * @return a matrix with predicted amount in each compartment 
 *         at each event.
 *
 * FIX ME: currently have a dummy msgs argument. Makes it easier
 * to expose to stan grammar files, because I can follow more closely
 * what was done for the ODE integrator. Not ideal.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                    const int nCmt,
                    const std::vector<T0>& time,
                    const std::vector<T1>& amt,
                    const std::vector<T2>& rate,
                    const std::vector<T3>& ii,
                    const std::vector<int>& evid,
                    const std::vector<int>& cmt,
                    const std::vector<int>& addl,
                    const std::vector<int>& ss,
                    const std::vector<std::vector<T4> >& pMatrix,
                    const std::vector<std::vector<T5> >& biovar,
                    const std::vector<std::vector<T6> >& tlag,
                    std::ostream* msgs = 0,
                    double rel_tol = 1e-10,
                    double abs_tol = 1e-10,
                    long int max_num_steps = 1e8) {  // NOLINT(runtime/int)
  using std::vector;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using boost::math::tools::promote_args;
  using refactor::PKRec;

  // check arguments
  static const char* function("generalOdeModel_bdf");
  torsten::pmetricsCheck(time, amt, rate, ii, evid, cmt, addl, ss,
                pMatrix, biovar, tlag, function);

  // Construct dummy matrix for last argument of pred
  Matrix<T4, Dynamic, Dynamic> dummy_system;
  vector<Matrix<T4, Dynamic, Dynamic> >
    dummy_systems(1, dummy_system);

  typedef general_functor<F> F0;

  const Pred1_general<F0> pred1(F0(f), rel_tol, abs_tol,
                                max_num_steps, msgs, "bdf");
  const PredSS_general<F0> predss (F0(f), rel_tol, abs_tol,
                                   max_num_steps, msgs, "bdf", nCmt);

#ifdef OLD_TORSTEN
  return Pred(time, amt, rate, ii, evid, cmt, addl, ss,
              pMatrix, biovar, tlag, nCmt, dummy_systems,
              pred1, predss);
#else
  using EM = EventsManager<T0, T1, T2, T3, T4, T5, T6>;
  EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);

  Matrix<typename EM::T_scalar, Dynamic, Dynamic> pred =
    Matrix<typename EM::T_scalar, Dynamic, Dynamic>::Zero(em.nKeep, nCmt);

  using model_type = refactor::PKODEModel<typename EM::T_time, typename EM::T_scalar, typename EM::T_rate, typename EM::T_par, F>;
  PkOdeIntegrator<PkBdf> integrator(rel_tol, abs_tol, max_num_steps, msgs);
  PredWrapper<model_type, PkOdeIntegrator<PkBdf>&> pr;
  pr.pred(em, pred, integrator, f);
  return pred;

#endif

}

/**
 * Overload function to allow user to pass an std::vector for 
 * pMatrix.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<T4>& pMatrix,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<std::vector<T6> >& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              vec_pMatrix, biovar, tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

/**
* Overload function to allow user to pass an std::vector for 
* pMatrix and biovar.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<T4>& pMatrix,
                     const std::vector<T5>& biovar,
                     const std::vector<std::vector<T6> >& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);
  std::vector<std::vector<T5> > vec_biovar(1, biovar);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              vec_pMatrix, vec_biovar, tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

/**
* Overload function to allow user to pass an std::vector for 
* pMatrix, biovar, and tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<T4>& pMatrix,
                     const std::vector<T5>& biovar,
                     const std::vector<T6>& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);
  std::vector<std::vector<T5> > vec_biovar(1, biovar);
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              vec_pMatrix, vec_biovar, vec_tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

/**
* Overload function to allow user to pass an std::vector for 
* pMatrix and tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<T4>& pMatrix,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<T6>& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T4> > vec_pMatrix(1, pMatrix);
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              vec_pMatrix, biovar, vec_tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

/**
* Overload function to allow user to pass an std::vector for 
* biovar.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<std::vector<T4> >& pMatrix,
                     const std::vector<T5>& biovar,
                     const std::vector<std::vector<T6> >& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T5> > vec_biovar(1, biovar);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, vec_biovar, tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

/**
* Overload function to allow user to pass an std::vector for 
* biovar and tlag.
*/
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<std::vector<T4> >& pMatrix,
                     const std::vector<T5>& biovar,
                     const std::vector<T6>& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T5> > vec_biovar(1, biovar);
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, vec_biovar, vec_tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

/**
 * Overload function to allow user to pass an std::vector for 
 * tlag.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
Eigen::Matrix <typename boost::math::tools::promote_args<T0, T1, T2, T3,
  typename boost::math::tools::promote_args<T4, T5, T6>::type>::type,
  Eigen::Dynamic, Eigen::Dynamic>
generalOdeModel_bdf(const F& f,
                     const int nCmt,
                     const std::vector<T0>& time,
                     const std::vector<T1>& amt,
                     const std::vector<T2>& rate,
                     const std::vector<T3>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<std::vector<T4> >& pMatrix,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<T6>& tlag,
                     std::ostream* msgs = 0,
                     double rel_tol = 1e-6,
                     double abs_tol = 1e-6,
                     long int max_num_steps = 1e6) {  // NOLINT(runtime/int)
  std::vector<std::vector<T6> > vec_tlag(1, tlag);

  return generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, biovar, vec_tlag,
                              msgs, rel_tol, abs_tol, max_num_steps);
}

  /*
   * For population models, more often we use ragged arrays
   * to describe the entire population, so in addition we need the arrays of
   * the length of each individual's data. The size of that
   * vector is the size of
   * the population.
   */
template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename F>
std::vector<Eigen::Matrix<typename EventsManager<T0, T1, T2, T3, T4, T5, T6>::T_scalar, // NOLINT
                          Eigen::Dynamic, Eigen::Dynamic> >
pop_pk_generalOdeModel_bdf(const F& f,
                           const int nCmt,
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
                           std::ostream* msgs = 0,
                           double rel_tol = 1e-10,
                           double abs_tol = 1e-10,
                           long int max_num_steps = 1e8) {
  using stan::math::check_consistent_sizes;
  using stan::math::check_greater_or_equal;

  int np = len.size();
  static const char* caller("generalOdeModel_bdf");
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

  using model_type = refactor::PKODEModel<typename EM::T_time, typename EM::T_scalar, typename EM::T_rate, typename EM::T_par, F>;
  PkOdeIntegrator<PkBdf> integrator(rel_tol, abs_tol, max_num_steps, msgs);
  PredWrapper<model_type, PkOdeIntegrator<PkBdf>&> pr;

  std::vector<Eigen::Matrix<typename EM::T_scalar, -1, -1>> pred(np);

  pr.pred(nCmt, len, time, amt, rate, ii, evid, cmt, addl, ss,
          len_pMatrix, pMatrix,
          len_biovar, biovar,
          len_tlag, tlag,
          pred, integrator, f);

  return pred;
}

}

#endif
