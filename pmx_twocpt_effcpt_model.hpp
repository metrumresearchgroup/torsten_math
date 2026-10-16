#ifndef STAN_MATH_TORSTEN_TWOCPT_EFFCPT_MODEL_HPP
#define STAN_MATH_TORSTEN_TWOCPT_EFFCPT_MODEL_HPP

#include <stan/math/torsten/pmx_linode_model.hpp>
#include <stan/math/torsten/pmx_cpt_disposition.hpp>
#include <stan/math/rev/fun/sqrt.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/torsten/PKModel/functors/check_mti.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_integrator.hpp>
#include <stan/math/torsten/dsolve/pk_vars.hpp>
#include <stan/math/torsten/pk_nvars.hpp>
#include <stan/math/prim/err/check_positive_finite.hpp>
#include <stan/math/prim/err/check_finite.hpp>
#include <stan/math/prim/err/check_nonnegative.hpp>

namespace torsten {
  /**
   * one-compartment model coupled with an effective compartment PK model.
   */
  struct PMXTwocptEffCptODE {
  /**
   * standard two compartment PK ODE RHS function
   * @tparam T0 t type
   * @tparam T1 initial condition type
   * @tparam T2 parameter type
   * @tparam T3 real data/rate type
   * @param t time
   * @param x initial condition
   * @param parms parameters
   * @param rate dosing rate
   * @param x_r real data
   * @param x_i integer data
   */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename stan::return_type<T0, T1, T2, T3>::type>
    operator()(const T0& t,
               const std::vector<T1>& x,
               const std::vector<T2>& parms,
               const std::vector<T3>& x_r,
               const std::vector<int>& x_i, std::ostream* pstream__) const {
      typedef typename stan::return_type<T0, T1, T2, T3>::type scalar;

      scalar
        CL = parms.at(0),
        Q = parms.at(1),
        V1 = parms.at(2),
        V2 = parms.at(3),
        ka = parms.at(4),
        ke = parms.at(5),
        k10 = CL / V1,
        k12 = Q / V1,
        k21 = Q / V2;

      std::vector<scalar> y(4, 0);
      y.at(0) = -ka * x.at(0);
      y.at(1) = ka * x.at(0) - (k10 + k12) * x.at(1) + k21 * x.at(2);
      y.at(2) = k12 * x.at(1) - k21 * x.at(2);
      y.at(3) = ke * x.at(1) - ke * x.at(3);

      return y;
    }

  /**
   * Eigen version where the statement variable is an Eigen vector.
   */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    Eigen::Matrix<typename stan::return_type_t<T0, T1, T2, T3>, -1, 1>
    operator()(const T0& t,
               const Eigen::Matrix<T1, -1, 1> & x,
               std::ostream* pstream__,
               const std::vector<T2>& parms,
               const std::vector<T3>& x_r,
               const std::vector<int>& x_i) const {
      typedef typename stan::return_type_t<T0, T1, T2, T3> scalar;

      T2 CL = parms.at(0);
      T2 Q  = parms.at(1);
      T2 V1 = parms.at(2);
      T2 V2 = parms.at(3);
      T2 ka = parms.at(4);
      T2 ke = parms.at(5);
      T2 k10 = CL / V1;
      T2 k12 = Q / V1;
      T2 k21 = Q / V2;

      Eigen::Matrix<scalar, -1, 1> y(4);
      y << -ka * x(0),
        ka * x(0) - (k10 + k12) * x(1) + k21 * x(2),
        k12 * x(1) - k21 * x(2),
        ke * x(1) - ke * x(3);

      return y;
    }
  };

  /**
   * twocpt-effcpt coupled model coupled with one effective compartment
   * model for PD effects. The static memebers provide
   * universal information: nb. of compartments,
   * nb. of parameters, and the RHS functor.
   *
   * @tparam T_par PK parameters type
   */
  template<typename T_par>
  class PMXTwocptEffCptModel {
    const T_par &CL_;
    const T_par &Q_;
    const T_par &V2_;
    const T_par &V3_;
    const T_par &ka_;
    const T_par &ke_;
    const T_par k10_;
    const T_par k12_;
    const T_par k21_;
    const std::vector<T_par> par_;
    const internal::PMXEffCptDisposition<internal::PMXTwoCptDisposition<T_par>> disp_;

  public:
    static constexpr int Ncmt = 4;
    static constexpr int Npar = 6;
    static constexpr PMXTwocptEffCptODE f_ = PMXTwocptEffCptODE();

    using par_type    = T_par;

  /**
   * Constructor
   *
   * @param par model parameters
   * @param CL clearance
   * @param V central cpt vol
   * @param ka absorption
   * @param ke effect cpt
   */
    PMXTwocptEffCptModel(const T_par& CL,
                         const T_par& Q,
                         const T_par& V2,
                         const T_par& V3,
                         const T_par& ka,
                         const T_par& ke) :
      CL_(CL),
      Q_(Q),
      V2_(V2),
      V3_(V3),
      ka_(ka),
      ke_(ke),
      k10_(CL_ / V2_),
      k12_(Q_ / V2_),
      k21_(Q_ / V3_),
      par_{CL_, Q_, V2_, V3_, ka_, ke_},
      disp_{{ka_, k10_, k12_, k21_}, ke_}
    {
      const char* fun = "PMXTwocptEffCptModel";
      stan::math::check_positive_finite(fun, "CL", CL_);
      stan::math::check_positive_finite(fun, "Q", Q_);
      stan::math::check_positive_finite(fun, "V2", V2_);
      stan::math::check_positive_finite(fun, "V3", V3_);
      stan::math::check_nonnegative(fun, "ka", ka_);
      stan::math::check_finite(fun, "ka", ka_);
      stan::math::check_positive_finite(fun, "ke", ke_);
    }

  /**
   * constructor
   *
   * @param par model parameters
   */
    PMXTwocptEffCptModel(const std::vector<T_par> & par) :
      PMXTwocptEffCptModel(par[0], par[1], par[2], par[3], par[4], par[5])
    {}

    /**
     * Get methods
     */
    const std::vector<T_par>  & par()     const { return par_;   }
    const PMXTwocptEffCptODE  & f()       const { return f_;     }
    const int                 & ncmt ()   const { return Ncmt;   }
    const int                 & npar ()   const { return Npar;   }

    Eigen::Matrix<T_par, -1, -1> to_linode_par() const {
      Eigen::Matrix<T_par, -1, -1> linode_par(Ncmt, Ncmt);
      linode_par << -ka_, 0.0, 0.0, 0.0,
        ka_, -(k10_ + k12_), k21_, 0.0,
        0.0, k12_, -k21_, 0.0,
        0.0, ke_, 0.0, -ke_;
      return linode_par;
    }

  /**
   * closed-form solution in terms of divided differences, well defined
   * when ka or ke equals a disposition rate or ka == ke
   */
    template<typename Tt0, typename Tt1, typename T, typename T1>
    void solve(PKRec<T>& y,
               const Tt0& t0, const Tt1& t1,
               const std::vector<T1>& rate,
               const dsolve::PMXAnalyiticalIntegrator& integ) const {
      internal::pmx_cpt_solve(disp_, y, t1 - t0, rate);
    }

  /**
   * Solve two-cpt + effcpt model: analytical solution
   */
    template<typename Tt0, typename Tt1, typename T, typename T1>
    void solve(PKRec<T>& y,
               const Tt0& t0, const Tt1& t1,
               const std::vector<T1>& rate) const {
      const dsolve::PMXAnalyiticalIntegrator integ;
      solve(y, t0, t1, rate, integ);
    }

  /**
   * Steady state model. We have to consider
   * different scenarios: bolus/multiple truncated infusion/const infusion
   *
   * @tparam T_amt amt type
   * @tparam T_r rate type
   * @tparam T_ii dosing interval type
   * @param t0 start time for steady state events
   * @param amt dosing amount
   * @param ii dosing interval
   * @param rate infusion rate
   * @param cmt dosing compartment
   */
    template<typename T_amt, typename T_r, typename T_ii>
    Eigen::Matrix<typename stan::return_type<T_par, T_amt, T_r, T_ii>::type, -1, 1>
    solve(double t0, const T_amt& amt, const T_r& rate, const T_ii& ii, const int& cmt) const {
      stan::math::check_positive_finite("steady state two-cpt solver", "cmt", cmt);
      stan::math::check_less("steady state two-cpt solver", "cmt", cmt, 4);
      stan::math::check_positive_finite("steady state two-cpt solver", "ka", ka_);

      return internal::pmx_cpt_steady_state(disp_, amt, rate, ii, cmt);
    }

    /**
     * wrapper to fit @c PrepWrapper's call signature
     */
    template<typename T_amt, typename T_r, typename T_ii>
    Eigen::Matrix<typename stan::return_type<T_par, T_amt, T_r, T_ii>::type, -1, 1>
    solve(double t0, const T_amt& amt, const T_r& rate, const T_ii& ii, const int& cmt,
          const dsolve::PMXAnalyiticalIntegrator& integrator) const {
      return solve(t0, amt, rate, ii, cmt);
    }
  };

  template<typename T_par>
  constexpr int PMXTwocptEffCptModel<T_par>::Ncmt;

  template<typename T_par>
  constexpr int PMXTwocptEffCptModel<T_par>::Npar;

  template<typename T_par>
  constexpr PMXTwocptEffCptODE PMXTwocptEffCptModel<T_par>::f_;
}

#endif
