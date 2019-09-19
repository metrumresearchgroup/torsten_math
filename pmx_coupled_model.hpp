#ifndef STAN_MATH_TORSTEN_COUPLED_MODEL_HPP
#define STAN_MATH_TORSTEN_COUPLED_MODEL_HPP

#include <stan/math/torsten/pmx_ode_integrator.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_twocpt_model.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <stan/math/torsten/return_type.hpp>
#include <stan/math/torsten/PKModel/Pred/Pred1_oneCpt.hpp>
#include <stan/math/torsten/PKModel/Pred/Pred1_twoCpt.hpp>
#include <stan/math/torsten/PKModel/Pred/SS_system2.hpp>
#include <stan/math/torsten/PKModel/functors/check_mti.hpp>

namespace refactor {

  using torsten::return_t;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using torsten::PMXOdeIntegrator;

  template<template<typename...> class T>
  struct PredSelector;

  template<>
  struct PredSelector<PMXOneCptModel> {using type = torsten::Pred1_oneCpt;};

  template<>
  struct PredSelector<PMXTwoCptModel> {using type = torsten::Pred1_twoCpt;};

  /**
   * In a coupled model's ODE functor, we first solve the PK
   * model using analytical solution, then pass it to the
   * numerical integrator to solve the ODE model. This
   * requires adapting the coupled model's functor to an ODE
   * functor that can be passed to ODE integrators.
   * The fuctor returns the derivative of the base ODE system when
   * @c rate is param.
   * The base PK component is calculated analytically and passed into
   * ODE functor as arguments.
   *  
   * @param t time
   * @param y indepedent PD variable
   * @param theta adapted coupled ODE params in the order of
   *    - original ODE param (PK param followed by PD param)
   *    - PK rate
   *    - PD rate
   *    - initial condition of PK ODE
   * @param x_r real data in the order of
   *    - dim PK states
   *    - initial time of PK ODE
   * @param x_i null int data
   */
  template <template<typename...> class T_m, typename F0, typename T_rate>
  struct PMXOdeFunctorCouplingAdaptor {
    PMXOdeFunctorCouplingAdaptor() {}

    /**
     *  default case: rate is a parameter, stored in theta.
     *  Theta contains in this order: ODE parameters, rates, and
     *  initial base PK states.
     */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename torsten::return_t<T0, T1, T2, T3>::type>
    operator()(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& theta,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream_) const {
      using std::vector;
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      typedef typename torsten::return_t<T0, T1, T2, T3>::type
        scalar;
      typedef typename torsten::return_t<T0, T2, T3>::type
        T_pk;  // return object of fTwoCpt  doesn't depend on T1
      using T_pkmodel = T_m<double, double, double, double>;

      size_t nPK = T_pkmodel::Ncmt;              // number of base PK states (for rates and inits)
      size_t nPD = y.size();                     // number of other states
      size_t nODEparms = theta.size() - 2 * nPK - nPD; // number of ODE parameters

      vector<T2> thetaPK(theta.begin(), theta.begin() + T_pkmodel::Npar);
      vector<T2> ratePK(theta.begin() + nODEparms, theta.begin() + nODEparms + nPK);

      // The last elements of theta contain the initial base PK states
      refactor::PKRec<T2> init_pk(nPK);
      for (int i = 0; i < nPK; ++i) {
        init_pk(nPK - i - 1) = *(theta.rbegin() + i);
      }

      // Last element of x_r contains the initial time
      T_m<T0, T2, T2, T2> pkmodel(x_r.back(), init_pk, ratePK, thetaPK);
      vector<scalar> dydt = F0()(t, y, to_array_1d(pkmodel.solve(t)), theta, x_r, x_i, pstream_);

      for (size_t i = 0; i < dydt.size(); i++)
        dydt[i] += theta[nODEparms + nPK + i];

      return dydt;
    }

    /*
     * when solving coupled model with @c var rate, we
     * append rate and PK initial condition to
     * parameter vector. 
     * FIXME: spurious @c var parameters will be generated
     * if the original parameters are data.
     */
    template<typename T, typename T0>
    static std::vector<typename torsten::return_t<T, T0, T_rate>::type>
    adapted_param(const std::vector<T> &par, const std::vector<T_rate> &rate,
                  const refactor::PKRec<T0>& y0_pk) {
      std::vector<stan::math::var> theta;
      theta.reserve(par.size() + rate.size() + y0_pk.size());
      theta.insert(theta.end(), par.begin(), par.end());
      theta.insert(theta.end(), rate.begin(), rate.end());
      for (int i = 0; i < y0_pk.size(); ++i) {
        theta.push_back(y0_pk(i)); 
      }
      return theta;
    }

    /*
     * when solving coupled model with @c var rate, @c x_r
     * is filled with initial time
     */
    template<typename T0>
    static std::vector<double>
    adapted_x_r(const std::vector<T_rate> &rate, const refactor::PKRec<T0>& y0_pk, double t0) {
      return {t0};
    }
  };

  template <template<typename...> class T_m, typename F0>
  struct PMXOdeFunctorCouplingAdaptor<T_m, F0, double> {
    PMXOdeFunctorCouplingAdaptor() {}

    /**
     * Returns the derivative of the base ODE system when @c rate is data.
     * The base PK component is calculated analytically and passed into
     * ODE functor as arguments.
     *  
     * @param t time
     * @param y indepedent PD  variable
     * @param theta adapted coupled ODE params in the order of
     *    - original ODE param
     *    - initial condition of PK ODE
     * @param x_r real data in the order of
     *    - PK rate
     *    - PD rate
     *    - initial time of PK ODE
     * @param x_i null int data
     */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename torsten::return_t<T0, T1, T2, T3>::type>
    operator()(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& theta,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream_) const {
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      using scalar = typename torsten::return_t<T0, T1, T2, T3>::type;
      using T_pk = typename torsten::return_t<T0, T2, T3>::type;
      using T_pkmodel = T_m<double, double, double, double>;

      // Get initial PK states stored at the end of @c theta
      int nPK = T_pkmodel::Ncmt;
      refactor::PKRec<T2> init_pk(nPK);
      for (int i = 0; i < nPK; ++i) {
        init_pk(nPK - i - 1) = *(theta.rbegin() + i);
      }

      // The last element of x_r contains the initial time,
      // and the beginning of theta are for PK params.
      T_m<T0, T2, T3, T2> pkmodel(x_r.back(), init_pk, x_r, theta);

      // move PK RHS to current time then feed the solution to PD ODE
      std::vector<T_pk> y_pk = to_array_1d(pkmodel.solve(t));
      std::vector<scalar> dydt = F0()(t, y, y_pk, theta, x_r, x_i, pstream_);

      // x_r: {pk rate, pd rate, t0}
      for (size_t i = 0; i < dydt.size(); ++i) {
        dydt[i] += x_r[nPK + i];
      }

      return dydt;
    }

    /*
     * when solving coupled model with @c var rate, we
     * append rate and PK initial condition to
     * parameter vector. 
     * FIXME: spurious @c var parameters will be generated
     * if the original parameters are data.
     */
    template<typename T, typename T0>
    static std::vector<typename torsten::return_t<T, T0>::type>
    adapted_param(const std::vector<T> &par, const std::vector<double> &rate,
                  const refactor::PKRec<T0>& y0_pk) {
      std::vector<typename torsten::return_t<T, T0>::type> theta;
      theta.reserve(par.size() + y0_pk.size());
      theta.insert(theta.end(), par.begin(), par.end());
      for (int i = 0; i < y0_pk.size(); ++i) {
        theta.push_back(y0_pk(i)); 
      }
      return theta;
    }

    /*
     * when solving coupled model with @c var rate, @c x_r
     * is filled with initial time
     */
    template<typename T0>
    static std::vector<double>
    adapted_x_r(const std::vector<double> &rate, const refactor::PKRec<T0>& y0_pk, double t0) {
      std::vector<double> res(rate);
      res.push_back(t0);
      return res;
    }
  };

  /**
   * A structure to store the algebraic system
   * which gets solved when computing the steady
   * state solution for coupled ODE models.
   *

   * @tparam T_amt @c amt type
   * @tparam T_rate @c rate type
   * @tparam F ODE RHS functor type
   * @tparam F2 type of the ODE that has analytical solution
   * @tparam T_integrator integrator type
   */
  template <typename T_amt, typename T_rate, typename T_ii, typename F, typename F2, typename T_integrator>
  struct PMXOdeFunctorCouplingSSAdaptor;

  /**
   * A structure to store the algebraic system
   * which gets solved when computing the steady
   * state solution.
   * 
   * In this structure, both amt and rate are fixed
   * variables.
   */
  template <typename F, typename F2, typename T_integrator>
  struct PMXOdeFunctorCouplingSSAdaptor<double, double, double, F, F2, T_integrator> {
    F f_;
    F2 f2_;
    double ii_;
    int cmt_;  // dosing compartment
    const T_integrator integrator_;
    int nPK_;

    PMXOdeFunctorCouplingSSAdaptor() {}

    PMXOdeFunctorCouplingSSAdaptor(const F& f,
                                   const F2& f2,
                                   double ii,
                                   int cmt,
                                   const T_integrator& integrator)
      : f_(f), f2_(f2), ii_(ii), cmt_(cmt), integrator_(integrator),
        nPK_(0) { }

    PMXOdeFunctorCouplingSSAdaptor(const F& f,
                                   const F2& f2,
                                   double ii,
                                   int cmt,
                                   const T_integrator& integrator,
                                   int nPK)
      : f_(f), f2_(f2), ii_(ii), cmt_(cmt), integrator_(integrator),
        nPK_(nPK) { }

    /**
     *  dd regime.
     *  dat contains the rates in each compartment followed
     *  by the adjusted amount (biovar * amt).
     */
    template <typename T0, typename T1>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T0, T1>::type,
                  Eigen::Dynamic, 1>
    operator()(const Eigen::Matrix<T0, Eigen::Dynamic, 1>& x,
               const Eigen::Matrix<T1, Eigen::Dynamic, 1>& y,
               const std::vector<double>& dat,
               const std::vector<int>& x_i,
               std::ostream* msgs) const {
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      using stan::math::to_vector;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;

      typedef typename boost::math::tools::promote_args<T0, T1>::type scalar;
      typedef typename stan::return_type<T0, T1>::type T_deriv;

      double t0 = 0;
      vector<double> ts(1);

      vector<scalar> x0(x.size());
      for (size_t i = 0; i < x0.size(); i++) x0[i] = x(i);
      double amt = dat[dat.size() - 1];
      double rate = dat[cmt_ - 1];

      // real data, which gets passed to the integrator, shoud not have
      // amt in it. Important for the mixed solver where the last element
      // is expected to be the absolute time (in this case, 0).
      vector<double> dat_ode = dat;
      dat_ode.pop_back();

      Eigen::Matrix<scalar, Eigen::Dynamic, 1> result(x.size());

      if (rate == 0) {  // bolus dose
        if ((cmt_ - nPK_) >= 0) x0[cmt_ - nPK_ - 1] += amt;

        vector<scalar> pred = integrator_(f_, x0, t0, ii_, to_array_1d(y),
                                          dat_ode, x_i)[0];

        for (int i = 0; i < result.size(); i++)
          result(i) = x(i) - pred[i];
      } else if (ii_ > 0) {  // multiple truncated infusions
        double delta = amt / rate;

        static const char* function("Steady State Event");
        torsten::check_mti(amt, delta, ii_, function);

        vector<scalar> pred;
        ts[0] = delta;  // time at which infusion stops
        x0 = integrator_(f_, to_array_1d(x), t0, ts, to_array_1d(y),
                         dat_ode, x_i)[0];

        Matrix<T1, Dynamic, 1> y2(y.size());
        int nParms = y.size() - nPK_;
        for (int i = 0; i < nParms; i++) y2(i) = y(i);

        if (nPK_ != 0) {
          Matrix<T1, 1, Dynamic> x0_pk(nPK_);
          for (int i = 0; i < nPK_; i++) x0_pk(i) = y(nParms + i);

          Matrix<T1, 1, Dynamic>
            x_pk = f2_(delta,
                       torsten::ModelParameters<double, T1, double, double>
                       (0, to_array_1d(y), vector<double>(),
                        vector<double>()),
                       x0_pk, dat_ode);

          for (int i = 0; i < nPK_; i++) y2(nParms + i) = x_pk(i);
        }

        ts[0] = ii_ - delta;
        dat_ode[cmt_ - 1] = 0;
        pred = integrator_(f_, x0, t0, ts, to_array_1d(y2), dat_ode, x_i)[0];

        for (int i = 0; i < result.size(); i++)
          result(i) = x(i) - pred[i];
      } else {  // constant infusion
        vector<T_deriv> derivative = f_(0, to_array_1d(x), to_array_1d(y),
                                        dat_ode, x_i, 0);
        result = to_vector(derivative);
      }

      return result;
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
   *             @c PMXOneCptmodel, @c PMXTwoCptmodel, @c PMXLinODEModel.
   * @tparam T_time type of time
   * @tparam T_init type of initial condition
   * @tparam T_rate type of dosing rate.
   * @tparam T_par type of parameter.
   * @tparam F type of ODE functor for @c PKODEModel.
   */
  template <template<typename...> class T_m,
            typename T_time, typename T_init, typename T_rate, typename T_par, typename F> // NOLINT
  class PKCoupledModel<T_m<T_time, T_init, T_rate, T_par>,
                       PKODEModel<T_time, T_init, T_rate, T_par,
                                  PMXOdeFunctorCouplingAdaptor<T_m, F, T_rate>> > { // NOLINT
    const refactor::PKRec<T_init>& y0_;
    const refactor::PKRec<T_init> y0_pk;
    const refactor::PKRec<T_init> y0_ode;
    PMXOdeFunctorCouplingAdaptor<T_m, F, T_rate> f;

  public:
    using Fa = PMXOdeFunctorCouplingAdaptor<T_m, F, T_rate>;
    const T_m<T_time, T_init, T_rate, T_par> pk_model;
    const PKODEModel<T_time, T_init, T_rate, T_par, Fa> ode_model;

    using pk_scalar_type = torsten::scalar_t<T_m<T_time, T_init, T_rate, T_par>>; // NOLINT 
    using ode_scalar_type = torsten::scalar_t<PKODEModel<T_time, T_init, T_rate, T_par, Fa> >; // NOLINT
    using scalar_type = typename stan::return_type<pk_scalar_type, ode_scalar_type>::type; // NOLINT 
    using init_type   = T_init;
    using time_type   = T_time;
    using par_type    = T_par;
    using rate_type   = T_rate;

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
                   const int n_ode) :
      y0_(init),
      y0_pk{ y0_.head(y0_.size() - n_ode) },
      y0_ode{ y0_.segment(y0_pk.size(), n_ode) },
      f(),
      pk_model(t0, y0_pk, rate, par),
      ode_model(t0, y0_ode, rate, par, f)
    {}
    
  private:

    /**
     * integrate coupled ODE.
     */
  template<typename T_r, typename T_integrator>
  refactor::PKRec<typename torsten::return_t<T_time, T_r, T_par, T_init>::type> // NOLINT
  integrate(const T_time& t_next,
            const std::vector<T_r>& rate,
            const T_integrator& integrator) const {
    using std::vector;
    using stan::math::to_array_1d;
    using torsten::return_t;

    typedef typename promote_args<T_time, T_r, T_par, T_init>::type scalar;

    // pass fixed times to the integrator. FIX ME - see issue #30
    T_time t0 = ode_model.t0();
    T_time t = t_next;
    vector<double> t_dbl{stan::math::value_of(t)};
    double t0_dbl = stan::math::value_of(t0);

    refactor::PKRec<scalar> pred;
    if (t_dbl[0] == t0_dbl) {
      pred = y0_;
    } else {
      size_t nPK = pk_model.ncmt();
      refactor::PKRec<scalar> xPK = pk_model.solve(t);

      // create vector with PD initial states
      vector<T_init> y0_PD(to_array_1d(y0_ode));
      PMXOdeFunctorCouplingAdaptor<T_m, F, T_r> f_coupled;
      vector<vector<scalar> >
        pred_V = integrator(f_coupled, y0_PD, t0_dbl, t_dbl,
                            f_coupled.adapted_param(ode_model.par(), rate, y0_pk),
                            f_coupled.adapted_x_r(rate, y0_pk, t0_dbl),
                            vector<int>());

      size_t nOde = pred_V[0].size();
      pred.resize(nPK + nOde);
      for (size_t i = 0; i < nPK; i++) pred(i) = xPK(i);
      for (size_t i = 0; i < nOde; i++) pred(nPK + i) = pred_V[0][i];
    }
    return pred;
  }

  /**
   * Steady state solver to
   * calculate the amount in each compartment at the end of a
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
  refactor::PKRec<typename torsten::return_t<T_ii, T_par>::type>
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

    typedef typename torsten::return_t<T_ii, T_par>::type scalar;

    double ii_dbl = stan::math::value_of(ii);

    // Compute solution for base 1cpt PK
    Matrix<T_par, Dynamic, 1> predPK;
    std::vector<T_par> pkpar = ode_model.par();
    int nPK = pk_model.ncmt();
    int nPD = ode_model.ncmt();
    const double t0 = 0.0;
    if (cmt <= nPK) {  // check dosing occurs in a base state
      T_m<double, double, double, T_par> pkmodel(t0, refactor::PKRec<double>(), std::vector<double>(), pk_model.par());
      predPK = pkmodel.solve(amt, rate, ii_dbl, cmt);
    } else {
      predPK = Matrix<scalar, Dynamic, 1>::Zero(nPK);
    }

    // Arguments for ODE integrator (and initial guess)
    std::vector<double> init_pd(nPD, 0.0);
    vector<double> x_r(nPK + nPD, 0);  // rate for the full system
    x_r.push_back(t0);  // include initial time (at SS, t0 = 0)
    vector<int> x_i;

    // Tuning parameters for algebraic solver
    double rel_tol = 1e-10;  // default
    double f_tol = 1e-4;  // empirical
    long int max_num_steps = 1e3;  // default // NOLINT

    using F_c = PMXOdeFunctorCouplingAdaptor<T_m, F, double>;
    F_c f_coupled;

    // construct algebraic system functor: note we adjust cmt
    // such that 1 corresponds to the first state we compute
    // numerically.
    using T_pred = typename PredSelector<T_m>::type;
    PMXOdeFunctorCouplingSSAdaptor<double, double, T_ii, F_c, T_pred, T_integrator >
      system(f_coupled, T_pred(), ii_dbl, cmt,
             integrator, nPK);

    Matrix<double, Dynamic, 1> predPD_guess;
    Matrix<scalar, 1, Dynamic> predPD;

    if (rate == 0) {  // bolus dose
      if (cmt > nPK) {
        init_pd[cmt - 1] = amt; 
      } else {
        predPK(cmt - 1) += amt;        
      }

      pkpar.insert(pkpar.end(), predPK.data(), predPK.data() + predPK.size());
      predPD_guess = to_vector(integrator(f_coupled, init_pd,
                                          0.0, ii_dbl,
                                          stan::math::value_of(pkpar),
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
                                         init_pd,
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
                                         init_pd,
                                         0.0, std::vector<double>(1, 100),
                                         torsten::unpromote(pkpar),
                                         x_r, x_i)[0]);

      x_r.push_back(amt);
      predPD = algebra_solver(system, predPD_guess,
                              to_vector(pkpar),
                              x_r, x_i,
                              0, rel_tol, f_tol, max_num_steps);
    }

    refactor::PKRec<scalar> pred(nPK + nPD);
    for (int i = 0; i < nPK; i++) pred(i) = predPK(i);
    for (int i = 0; i < nPD; i++) pred(nPK + i) = predPD(i);

    return pred;
  }

  /**
   * Case 2 (vd): amt is random, rate is fixed.
   */
  template<typename T_ii, typename T_amt, typename T_integrator>
  refactor::PKRec<typename stan::return_type<T_ii, T_amt, T_par>::type>
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
    using stan::math::value_of;

    typedef typename torsten::return_t<T_ii, T_amt,
      T_par>::type scalar;

    double ii_dbl = value_of(ii);

    // Compute solution for base 1cpt PK
    Matrix<scalar, Dynamic, 1> predPK;
    std::vector<T_par> pkpar = ode_model.par();
    int nPK = pk_model.ncmt();
    int nPD = ode_model.ncmt();
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
    Matrix<double, 1, Dynamic> init_pd
      = Matrix<double, 1, Dynamic>::Zero(nPD);
    vector<double> x_r(nPK + nPD, 0);  // rate for the full system
    x_r.push_back(0);  // include initial time (at SS, t0 = 0)
    vector<int> x_i;

    // Tuning parameters for algebraic solver
    double rel_tol = 1e-10;  // default
    double f_tol = 1e-4;  // empirical
    long int max_num_steps = 1e3;  // default // NOLINT

    using F_c = PMXOdeFunctorCouplingAdaptor<T_m, F, double>;
    F_c f_coupled;

    torsten::SS_system2_vd<F_c, T_integrator >
      system(f_coupled, ii_dbl, cmt, integrator, nPK);

    Matrix<double, Dynamic, 1> predPD_guess;
    Matrix<scalar, 1, Dynamic> predPD;

    if (rate == 0) {  // bolus dose
      predPD_guess = to_vector(integrator(f_coupled,
                                        to_array_1d(init_pd),
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
                                         to_array_1d(init_pd),
                                         0.0, std::vector<double>(1, 100),
                                         torsten::unpromote(theta2),
                                         x_r, x_i)[0]);

      predPD = algebra_solver(system, predPD_guess,
                              to_vector(theta2),
                              x_r, x_i,
                              0, rel_tol, f_tol, max_num_steps);
    }

    Matrix<scalar, Dynamic, 1> pred(nPK + nPD);
    for (int i = 0; i < nPK; i++) pred(i) = predPK(i);
    for (int i = 0; i < nPD; i++) pred(nPK + i) = predPD(i);

    return pred;
  }

  public:
    /* 
     * solve the coupled model.
     */
    template<PMXOdeIntegratorId It>
    refactor::PKRec<scalar_type>
    solve(const T_time& t_next,
          const PMXOdeIntegrator<It>& integrator) const {
      return integrate(t_next, ode_model.rate(), integrator);
    }

    /* 
     * solve the coupled model, steady state. We delegate
     * the solution to @c integrate, in which the type of @c
     * amt will be used for template partial specification.
     */
    template<PMXOdeIntegratorId It, typename T_ii, typename T_amt>
    refactor::PKRec<scalar_type>
    solve(const T_amt& amt, const double& rate, const T_ii& ii, const int& cmt,
          const PMXOdeIntegrator<It>& integrator) const {
      return integrate(amt, rate, ii, cmt, integrator);
    }
  };

  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F> // NOLINT
  using PkOneCptOdeModel =
    PKCoupledModel< PMXOneCptModel<T_time, T_init, T_rate, T_par>,
                    PKODEModel<T_time, T_init, T_rate, T_par,
                               PMXOdeFunctorCouplingAdaptor<PMXOneCptModel, F, T_rate>> >; // NOLINT

  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F> // NOLINT
  using PkTwoCptOdeModel =
    PKCoupledModel< PMXTwoCptModel<T_time, T_init, T_rate, T_par>,
                    PKODEModel<T_time, T_init, T_rate, T_par,
                               PMXOdeFunctorCouplingAdaptor<PMXTwoCptModel, F, T_rate>> >; // NOLINT
}


#endif
