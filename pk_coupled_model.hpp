#ifndef STAN_MATH_TORSTEN_COUPLED_MODEL_HPP
#define STAN_MATH_TORSTEN_COUPLED_MODEL_HPP

#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <stan/math/torsten/pk_ode_model.hpp>
#include <stan/math/torsten/PKModel/Pred/Pred1_oneCpt.hpp>
#include <stan/math/torsten/PKModel/Pred/Pred1_twoCpt.hpp>
#include <stan/math/torsten/PKModel/Pred/SS_system2.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  template<template<typename...> class T>
  struct PredSelector;

  template<>
  struct PredSelector<PKOneCptModel> {using type = torsten::Pred1_oneCpt;};

  template<>
  struct PredSelector<PKTwoCptModel> {using type = torsten::Pred1_twoCpt;};

  /**
   * In a coupled model's ODE functor, we first solve the PK
   * model using analytical solution, then pass it to the
   * numerical integrator to solve the ODE model. This
   * requires adapting the coupled model's functor to an ODE
   * functor that can be passed to ODE integrators.
   *
   * @tparam T_m (PK) model that will be solved analytically.
   * @tparam F0 original ODE functor
   * @tparam T_rate type of dosing rate.
   */
  template <template<typename...> class T_m, typename F0, typename T_rate>
  struct PKOdeFunctorCouplingAdaptor {
    F0 f0_;

    PKOdeFunctorCouplingAdaptor() { }

    explicit PKOdeFunctorCouplingAdaptor(const F0& f0) : f0_(f0) { }

    /**
     *  Case 2: rate is a parameter, stored in theta.
     *  Theta contains in this order: ODE parameters, rates, and
     *  initial base PK states.
     */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
    operator()(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& theta,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream_) const {
      using std::vector;
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type
        scalar;
      typedef typename boost::math::tools::promote_args<T0, T2, T3>::type
        T_pk;  // return object of fTwoCpt  doesn't depend on T1
      using T_pkmodel = T_m<double, double, double, double>;

      size_t
        nTheta = theta.size(),
        nPK = T_pkmodel::Ncmt,  // number of base PK states (equivalently rates and inits)
        nPD = y.size(),  // number of other states
        nODEparms = nTheta - 2 * nPK - nPD;  // number of ODE parameters

      // Theta first contains the base PK parameters, followed by
      // the other ODE parameters.
      int nParmsPK = T_pkmodel::Npar;
      vector<T2> thetaPK(theta.begin(), theta.begin() + nParmsPK);

      // Next theta contains the rates for the base PK compartments.
      vector<T2> ratePK(nPK);
      for (size_t i = 0; i < nPK; i++)
        ratePK[i] = theta[nODEparms + i];

      // followed by the rates in the other compartments.
      vector<T2> ratePD(nPD);
      for (size_t i = 0; i < nPD; i++)
        ratePD[i] = theta[nODEparms + nPK + i];

      // The last elements of theta contain the initial base PK states
      refactor::PKRec<T2> init_pk(nPK);
      for (size_t i = 0; i < nPK; ++i) init_pk(i) = theta[nTheta - nPK + i];

      // Last element of x_r contains the initial time
      T0 dt = t - x_r[x_r.size() - 1];

      T0 t0 = x_r[x_r.size() - 1];
      T_m<T0, T2, T2, T2> pkmodel(t0, init_pk, ratePK, thetaPK);
      std::vector<T_pk> y_pk = to_array_1d(pkmodel.solve(dt));
      // vector<T_pk> y_pk = fOneCpt(dt, thetaPK, init_pk, ratePK);
      vector<scalar> dydt = f0_(dt, y, y_pk, theta, x_r, x_i, pstream_);

      for (size_t i = 0; i < dydt.size(); i++)
        dydt[i] += ratePD[i];

      return dydt;
    }
  };

  template <template<typename...> class T_m, typename F0>
  struct PKOdeFunctorCouplingAdaptor<T_m, F0, double> {
    F0 f0_;

    PKOdeFunctorCouplingAdaptor() { }

    explicit PKOdeFunctorCouplingAdaptor(const F0& f0) : f0_(f0) { }

    /**
     *  Returns the derivative of the base ODE system. The base 1 PK
     *  component is calculated analytically.
     *  
     *  theta stores the ODE parameters, followed by the two initial
     *  PK states.
     *  x_r stores the rates for the FULL system, followed by the initial
     *  time.
     *
     *  Case 1: rate is fixed data and is passed through x_r.
     */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
    operator()(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& theta,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream_) const {
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type
        scalar;
      typedef typename boost::math::tools::promote_args<T0, T2, T3>::type
        T_pk;  // return object of fOneCpt  doesn't depend on T1

      // Get PK parameters
      using T_pkmodel = T_m<double, double, double, double>;

      int nParmsPK = T_pkmodel::Npar;
      std::vector<T2> thetaPK(theta.begin(), theta.begin() + nParmsPK);

      // Get initial PK states
      int nPK = T_pkmodel::Ncmt;
      refactor::PKRec<T2> init_pk(nPK);
      size_t nTheta = theta.size();
      for (int i = 0; i < nPK; ++i) init_pk(i) = theta[nTheta - nPK + i];

      // The last element of x_r contains the absolutime
      T0 dt = t - x_r[x_r.size() - 1];

      T0 t0 = x_r[x_r.size() - 1];
      T_m<T0, T2, T3, T2> pkmodel(t0, init_pk, x_r, thetaPK);
      std::vector<T_pk> y_pk = to_array_1d(pkmodel.solve(dt));
      std::vector<scalar> dydt = f0_(dt, y, y_pk, theta, x_r, x_i, pstream_);

      for (size_t i = 0; i < dydt.size(); i++)
        dydt[i] += x_r[nPK + i];

      return dydt;
    }
  };

  /**
   * Coupled model.
   *
   * @tparam T_m1 type of the 1st model in the coupling.
   * @tparam T_m2 type of the 2nd model in the coupling.
   */
  template <typename T_m1, typename T_m2>
  class PKCoupledModel;

  /**
   * Specialization of coupled model with 2nd model
   * being @c PKODEModel.
   *
   * @tparam T_m type of 1st model, choose among
   *             @c PKOneCptmodel, @c PKTwoCptmodel, @c PKLinODEModel.
   * @tparam T_time type of time
   * @tparam T_init type of initial condition
   * @tparam T_rate type of dosing rate.
   * @tparam T_par type of parameter.
   * @tparam F type of ODE functor for @c PKODEModel.
   * @tparam Ti type of extra param for @c PKODEModel.
   */
  template <template<typename...> class T_m,
            typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti> // NOLINT
  class PKCoupledModel<T_m<T_time, T_init, T_rate, T_par>,
                       PKODEModel<T_time, T_init, T_rate, T_par,
                                  PKOdeFunctorCouplingAdaptor<T_m, F, T_rate>, Ti> > { // NOLINT
    const refactor::PKRec<T_init>& y0;
    const refactor::PKRec<T_init> y0_pk;
    const refactor::PKRec<T_init> y0_ode;
    const F& f_;
    PKOdeFunctorCouplingAdaptor<T_m, F, T_rate> f;

  public:
    using Fa = PKOdeFunctorCouplingAdaptor<T_m, F, T_rate>;
    const T_m<T_time, T_init, T_rate, T_par> pk_model;
    const PKODEModel<T_time, T_init, T_rate, T_par, Fa, Ti> ode_model;

    using pk_scalar_type = torsten::scalar_t<T_m<T_time, T_init, T_rate, T_par>>; // NOLINT 
    using ode_scalar_type = torsten::scalar_t<PKODEModel<T_time, T_init, T_rate, T_par, Fa, Ti> >; // NOLINT
    using scalar_type = typename stan::return_type<pk_scalar_type, ode_scalar_type>::type; // NOLINT 

    /*
     * FIX ME: we need to get rid of @c ModelParameters in
     * @c Pred2
     */
    template<typename T0, typename T1, typename T2, typename T3>
    static std::vector<T1>
    get_param(const torsten::ModelParameters<T0, T1, T2, T3>& p) {
      return p.get_RealParameters();
    }

    /**
     * Coupled model constructor
     *
     * @param t0 initial time
     * @param y0 initial condition, with PK model's initial
     *           condition followed by ODE model's initial condition.
     * @param rate dosing rate
     * @param par model parameters
     * @param f ODE functor
     * @param n_ode the size of ode_model's ODE system
     */
    PKCoupledModel(const T_time& t0,
                   const PKRec<T_init>& init,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const F& f0,
                   const Ti& n_ode) :
      y0(init),
      y0_pk{ y0.head(y0.size() - n_ode) },
      y0_ode{ y0.segment(y0_pk.size(), n_ode) },
      f_(f0),
      f(f0),
      pk_model({t0, y0_pk, rate, par}),
      ode_model({t0, y0_ode, rate, par, f, n_ode})
    {}
    
  private:

  template<typename T_integrator>
  Eigen::Matrix<typename boost::math::tools::promote_args<T_time,
    T_par, T_init>::type, Eigen::Dynamic, 1>
  integrate(  const T_time& dt,
              const std::vector<double>& rate,
              const T_integrator& integrator) const {
    using std::vector;
    using stan::math::to_array_1d;
    using boost::math::tools::promote_args;

    typedef typename promote_args<T_time, T_par, T_init>::type scalar;
    typedef typename promote_args<T_par, T_init>::type T_theta;

    auto parameter = ode_model.par();
    const refactor::PKRec<T_init>& init = y0;

    assert((size_t) init.cols() == rate.size());

    // pass fixed times to the integrator. FIX ME - see issue #30
    T_time t0 = ode_model.t0();  // time of previous event
    T_time t = t0 + dt;  // time of current event
    vector<double> t_dbl(1, torsten::unpromote(t));
    double t0_dbl = torsten::unpromote(t0);

    vector<double> x_r = rate;
    x_r.push_back(t0_dbl);  // need to pass the initial time!

    vector<scalar> y0 = to_array_1d(init);

    size_t nParm = parameter.size();
    vector<T_theta> theta(nParm);
    for (size_t i = 0; i < nParm; i++)
      theta[i] = parameter[i];

    Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;
    if (t_dbl[0] == t0_dbl) { pred = init;
    } else {
      size_t nPK = pk_model.ncmt();  // two states for 1Cpt with absorption

      refactor::PKRec<scalar> xPK = pk_model.solve(dt);

      // Add PK inits to theta
      for (size_t i = 0; i < nPK; i++) theta.push_back(init(i));

      // create vector with PD initial states
      size_t nPD = init.size() - nPK;
      vector<scalar> y0_PD(nPD);
      for (size_t i = 0; i < nPD; i++) y0_PD[i] = y0[nPK + i];
      vector<int> idummy;

      using F_c = PKOdeFunctorCouplingAdaptor<T_m, F, double>;
      F_c f_coupled(f_);
      vector<vector<scalar> >
        pred_V = integrator(f_coupled,
                             y0_PD, t0_dbl, t_dbl,
                             theta, x_r, idummy);
      size_t nOde = pred_V[0].size();

      pred.resize(nPK + nOde);
      for (size_t i = 0; i < nPK; i++) pred(i) = xPK(i);
      for (size_t i = 0; i < nOde; i++) pred(nPK + i) = pred_V[0][i];
    }
    return pred;
  }

  /**
  * Overload function for case rate is a vector of var.
  * The parameters are stored in theta in this order:
  *   (1) ODE parameters
  *   (2) rate
  *   (3) initial states for base PK
  *
  * Unlike the general solver (pred1_general_solver), we'll
  * call the mix_rate_var_functor, which will know where rate
  * is located inside theta.
  */
  template<typename T_r,
           typename T_integrator>
  Eigen::Matrix<typename boost::math::tools::promote_args<T_time,
    T_r, T_par, T_init>::type, Eigen::Dynamic, 1>
  integrate(  const T_time& dt,
              const std::vector<T_r>& rate,
              const T_integrator& integrator) const {
    using std::vector;
    using stan::math::to_array_1d;
    using boost::math::tools::promote_args;

    typedef typename promote_args<T_time, T_r, T_par,
                                  T_init>::type scalar;
    typedef typename promote_args<T_par, T_init, T_r>::type T_theta;

    auto parameter = ode_model.par();
    const refactor::PKRec<T_init>& init = y0;

    assert((size_t) init.cols() == rate.size());

    // pass fixed times to the integrator. FIX ME - see issue #30
    T_time t0 = ode_model.t0();
    T_time t = t0 + dt;

    vector<double> t_dbl(1, torsten::unpromote(t));
    double t0_dbl = torsten::unpromote(t0);

    vector<scalar> y0 = to_array_1d(init);

    size_t nParm = parameter.size();
    vector<T_theta> theta(nParm);
    for (size_t i = 0; i < nParm; i++)
      theta[i] = parameter[i];

    Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;
    if (t_dbl[0] == t0_dbl) { pred = init;
    } else {
      size_t nPK = pk_model.ncmt();  // two states for 1Cpt with absorption

      refactor::PKRec<scalar> xPK = pk_model.solve(dt);

      // Add rate and PK inits IN THIS ORDER to theta
      for (size_t i = 0; i < rate.size(); i++) theta.push_back(rate[i]);
      for (size_t i = 0; i < nPK; i++) theta.push_back(init(i));

      // create vector with PD initial states
      size_t nPD = init.size() - nPK;
      vector<scalar> y0_PD(nPD);
      for (size_t i = 0; i < nPD; i++) y0_PD[i] = y0[nPK + i];
      vector<int> idummy;

      vector<double> x_r(2);
      x_r[0] = nPK;
      x_r[1] = t0_dbl;

      using F_c = PKOdeFunctorCouplingAdaptor<T_m, F, T_r>;
      F_c f_coupled(f_);
      vector<vector<scalar> >
        pred_V = integrator(f_coupled,
                             y0_PD, t0_dbl, t_dbl,
                             theta, x_r, idummy);

      size_t nOde = pred_V[0].size();
      pred.resize(nPK + nOde);
      for (size_t i = 0; i < nPK; i++) pred(i) = xPK(i);
      for (size_t i = 0; i < nOde; i++) pred(nPK + i) = pred_V[0][i];
    }
    return pred;
  }

  /**
   * Mix1 compartment model using built-in ODE solver, mixed
   * solving method, and root-finder.
   * Calculate amount in each compartment at the end of a
   * steady-state dosing interval or during a steady-state
   * constant input (if ii = 0). The function is overloaded
   * to address the cases where amt or rate may be fixed or
   * random variables (yielding a total of 4 cases).
   * 
   * Case 1 (dd): amt and rate are fixed.
   *
   *	 @tparam T_time type of scalar for time
   *	 @tparam T_ii type of scalar for interdose interval
   *	 @tparam T_parameters type of scalar for ODE parameters
   *   @tparam T_biovar type of scalar for bio-availability
   *	 @tparam F type of ODE system function
   *	 @param[in] parameter model parameters at current event
   *	 @param[in] rate
   *	 @param[in] ii interdose interval
   *	 @param[in] cmt compartment in which the event occurs
   *	 @param[in] f functor for base ordinary differential equation
   *              that defines compartment model
   *   @return an eigen vector that contains predicted amount in each
   *           compartment at the current event.
   */
  template<typename T_ii, typename T_integrator>
  Eigen::Matrix<typename boost::math::tools::promote_args<T_ii,
    T_par>::type, Eigen::Dynamic, 1>
  integrate(const double& amt,
              const double& rate,
              const T_ii& ii,
              const int& cmt,
              const T_integrator& integrator) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using Eigen::VectorXd;
    using std::vector;
    using stan::math::algebra_solver;
    using stan::math::to_vector;
    using stan::math::to_array_1d;

    typedef typename boost::math::tools::promote_args<T_ii, T_par>::type scalar;

    double ii_dbl = torsten::unpromote(ii);

    // Compute solution for base 1cpt PK
    Matrix<T_par, Dynamic, 1> predPK;
    std::vector<T_par> pkpar = ode_model.par();
    int nPK = pk_model.ncmt();
    int nOde_ = ode_model.ncmt();
    if (cmt <= nPK) {  // check dosing occurs in a base state
      // PredSS_twoCpt PredSS_one;
      // int nParmsPK = 3;
      
      const double t0 = 0.0;
      const refactor::PKRec<double> y0;
      const std::vector<double> rate_dummy;
        
      T_m<double, double, double, T_par> pkmodel(t0, y0, rate_dummy, pk_model.par());
      predPK = pkmodel.solve(amt, (cmt <= nPK) ? rate : 0, ii_dbl, cmt);
    } else {
      predPK = Matrix<scalar, Dynamic, 1>::Zero(nPK);
    }

    // Arguments for ODE integrator (and initial guess)
    Matrix<double, 1, Dynamic> init_dbl
      = Matrix<double, 1, Dynamic>::Zero(nOde_);
    vector<double> x_r(nPK + nOde_, 0);  // rate for the full system
    x_r.push_back(0);  // include initial time (at SS, t0 = 0)
    vector<int> x_i;

    // Tuning parameters for algebraic solver
    double rel_tol = 1e-10;  // default
    double f_tol = 1e-4;  // empirical
    long int max_num_steps = 1e3;  // default // NOLINT

    using F_c = PKOdeFunctorCouplingAdaptor<T_m, F, double>;
    F_c f_coupled(f_);

    // construct algebraic system functor: note we adjust cmt
    // such that 1 corresponds to the first state we compute
    // numerically.
    using T_pred = typename PredSelector<T_m>::type;
    torsten::SS_system2_dd<F_c, T_pred, T_integrator >
      system(f_coupled, T_pred(), ii_dbl, cmt,
             integrator, nPK);

    Matrix<double, Dynamic, 1> predPD_guess;
    Matrix<scalar, 1, Dynamic> predPD;

    if (rate == 0) {  // bolus dose
      if (cmt > nPK) init_dbl(cmt - 1) = amt;
      else
        predPK(cmt - 1) += amt;

      pkpar.insert(pkpar.end(), predPK.data(),
                   predPK.data() + predPK.size());

      predPD_guess = to_vector(integrator(f_coupled,
                                        to_array_1d(init_dbl),
                                        0.0, std::vector<double>(1, ii_dbl),
                                        torsten::unpromote(pkpar),
                                        x_r, x_i)[0]);

      x_r.push_back(amt);
      predPD = algebra_solver(system, predPD_guess,
                              to_vector(pkpar),
                              x_r, x_i,
                              0, rel_tol, f_tol, max_num_steps);

      // Remove dose input in dosing compartment. Pred will add it
      // later, so we want to avoid redundancy.
      if (cmt <= nPK) predPK(cmt - 1) -= amt;

    } else if (ii > 0) {  // multiple truncated infusions
      x_r[cmt - 1] = rate;

      pkpar.insert(pkpar.end(), predPK.data(),
                   predPK.data() + predPK.size());
      predPD_guess = to_vector(integrator(f_coupled,
                                         to_array_1d(init_dbl),
                                         0.0, std::vector<double>(1, ii_dbl),
                                         torsten::unpromote(pkpar),
                                         x_r, x_i)[0]);

      x_r.push_back(amt);  // needed?
      predPD = algebra_solver(system, predPD_guess,
                            to_vector(pkpar),
                            x_r, x_i,
                            0, rel_tol, f_tol, max_num_steps);
    } else {  // constant infusion
      x_r[cmt - 1] = rate;

      pkpar.insert(pkpar.end(), predPK.data(),
                   predPK.data() + predPK.size());
      predPD_guess = to_vector(integrator(f_coupled,
                                         to_array_1d(init_dbl),
                                         0.0, std::vector<double>(1, 100),
                                         torsten::unpromote(pkpar),
                                         x_r, x_i)[0]);

      x_r.push_back(amt);
      predPD = algebra_solver(system, predPD_guess,
                              to_vector(pkpar),
                              x_r, x_i,
                              0, rel_tol, f_tol, max_num_steps);
    }

    Matrix<scalar, Dynamic, 1> pred(nPK + nOde_);
    for (int i = 0; i < nPK; i++) pred(i) = predPK(i);
    for (int i = 0; i < nOde_; i++) pred(nPK + i) = predPD(i);

    return pred;
  }

  /**
   * Case 2 (vd): amt is random, rate is fixed.
   */
  template<typename T_ii,
           typename T_amt,
           typename T_integrator>
  Eigen::Matrix<typename stan::return_type<T_ii, T_amt, T_par>::type, Eigen::Dynamic, 1>
  integrate(const T_amt& amt,
              const double& rate,
              const T_ii& ii,
              const int& cmt,
              const T_integrator& integrator) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using Eigen::VectorXd;
    using std::vector;
    using stan::math::algebra_solver;
    using stan::math::to_vector;
    using stan::math::to_array_1d;
    using stan::math::invalid_argument;

    typedef typename boost::math::tools::promote_args<T_ii, T_amt,
      T_par>::type scalar;

    double ii_dbl = torsten::unpromote(ii);

    // Compute solution for base 1cpt PK
    Matrix<scalar, Dynamic, 1> predPK;
    std::vector<T_par> pkpar = ode_model.par();
    int nPK = pk_model.ncmt();
    int nOde_ = ode_model.ncmt();
    if (cmt <= nPK) {  // check dosing occurs in a base state
      // PredSS_twoCpt PredSS_one;
      // int nParmsPK = 3;

      const double t0 = 0.0;
      const refactor::PKRec<double> y0;
      const std::vector<T_amt> rate_dummy;
      T_m<double, double, T_amt, T_par> pkmodel(t0, y0, rate_dummy, pk_model.par());
      predPK = pkmodel.solve(amt, rate, ii_dbl, cmt);
      predPK(cmt - 1) = predPK(cmt - 1) + amt;
    } else {
      predPK = Matrix<scalar, Dynamic, 1>::Zero(nPK);
    }

    std::vector<scalar> theta2;
    theta2.insert(theta2.end(), pkpar.begin(), pkpar.end());
    theta2.insert(theta2.end(), predPK.data(),
                  predPK.data() + predPK.size());
    theta2.push_back(amt);

    // Arguments for ODE integrator (and initial guess)
    Matrix<double, 1, Dynamic> init_dbl
      = Matrix<double, 1, Dynamic>::Zero(nOde_);
    vector<double> x_r(nPK + nOde_, 0);  // rate for the full system
    x_r.push_back(0);  // include initial time (at SS, t0 = 0)
    vector<int> x_i;

    // Tuning parameters for algebraic solver
    double rel_tol = 1e-10;  // default
    double f_tol = 1e-4;  // empirical
    long int max_num_steps = 1e3;  // default // NOLINT

    using F_c = PKOdeFunctorCouplingAdaptor<T_m, F, double>;
    F_c f_coupled(f_);

    torsten::SS_system2_vd<F_c, T_integrator >
      system(f_coupled, ii_dbl, cmt, integrator, nPK);

    Matrix<double, Dynamic, 1> predPD_guess;
    Matrix<scalar, 1, Dynamic> predPD;

    if (rate == 0) {  // bolus dose
      predPD_guess = to_vector(integrator(f_coupled,
                                        to_array_1d(init_dbl),
                                        0.0, std::vector<double>(1, ii_dbl),
                                        torsten::unpromote(theta2),
                                        x_r, x_i)[0]);

      predPD = algebra_solver(system, predPD_guess,
                              to_vector(theta2),
                              x_r, x_i,
                              0, rel_tol, f_tol, max_num_steps);

      if (cmt <= nPK) predPK(cmt - 1) -= amt;
    } else if (ii > 0) {
      invalid_argument("Steady State Event",
                       "Current version does not handle the case of",
                       "", " multiple truncated infusions ",
                       "(i.e ii > 0 and rate > 0) when F * amt is a parameter.");  // NOLINT
    } else {
      x_r[cmt - 1] = rate;

      predPD_guess = to_vector(integrator(f_coupled,
                                         to_array_1d(init_dbl),
                                         0.0, std::vector<double>(1, 100),
                                         torsten::unpromote(theta2),
                                         x_r, x_i)[0]);

      predPD = algebra_solver(system, predPD_guess,
                              to_vector(theta2),
                              x_r, x_i,
                              0, rel_tol, f_tol, max_num_steps);
    }

    Matrix<scalar, Dynamic, 1> pred(nPK + nOde_);
    for (int i = 0; i < nPK; i++) pred(i) = predPK(i);
    for (int i = 0; i < nOde_; i++) pred(nPK + i) = predPD(i);

    return pred;
  }

  public:
    /* 
     * solve the coupled model.
     */
    template<PkOdeIntegratorId It>
    Eigen::Matrix<scalar_type, Dynamic, 1>
    solve(const T_time& dt,
          const PkOdeIntegrator<It>& integrator) const {
      return integrate(dt, ode_model.rate(), integrator);
    }

    /* 
     * solve the coupled model, steady state. We delegate
     * the solution to @c integrate, in which the type of @c
     * amt will be used for template partial specification.
     */
    template<PkOdeIntegratorId It, typename T_ii, typename T_amt>
    Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
    solve(const T_amt& amt, const double& rate, const T_ii& ii, const int& cmt,
          const PkOdeIntegrator<It>& integrator) const {
      return integrate(amt, rate, ii, cmt, integrator);
    }
  };

  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti> // NOLINT
  using PkOneCptOdeModel =
    PKCoupledModel< PKOneCptModel<T_time, T_init, T_rate, T_par>,
                    PKODEModel<T_time, T_init, T_rate, T_par,
                               PKOdeFunctorCouplingAdaptor<PKOneCptModel, F, T_rate>, Ti> >; // NOLINT

  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti> // NOLINT
  using PkTwoCptOdeModel =
    PKCoupledModel< PKTwoCptModel<T_time, T_init, T_rate, T_par>,
                    PKODEModel<T_time, T_init, T_rate, T_par,
                               PKOdeFunctorCouplingAdaptor<PKTwoCptModel, F, T_rate>, Ti> >; // NOLINT
}


#endif
