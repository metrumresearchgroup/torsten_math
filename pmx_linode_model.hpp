#ifndef STAN_MATH_PMX_LINODE_MODEL_HPP
#define STAN_MATH_PMX_LINODE_MODEL_HPP

#include <stan/math/prim/fun/multiply.hpp>
#include <stan/math/rev/fun/multiply.hpp>
#include <stan/math/prim/fun/matrix_exp.hpp>
#include <stan/math/prim/fun/mdivide_left.hpp>
#include <stan/math/rev/fun/mdivide_left.hpp>
#include <stan/math/torsten/model_solve_d.hpp>
#include <stan/math/torsten/PKModel/functors/check_mti.hpp>
#include <stan/math/torsten/dsolve/pk_vars.hpp>

namespace torsten {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using Eigen::Map;

  struct PMXLinODE {
    /**
     * standard two compartment PK ODE RHS function
     * @tparam T0 t type
     * @tparam T1 initial condition type
     * @tparam T2 parameter type
     * @tparam T3 real data/rate type
     * @param t type
     * @param x initial condition type
     * @param parms parameters that form a n x n matrix, where
     * @c (n = x.size()). We assume the matrix is col-majored.
     * @param rate dosing rate
     * @param dummy dummy
     */
    template <typename T0, typename T1, typename T2>
    inline
    std::vector<typename stan::return_type<T0, T1, T2>::type>
    operator()(const T0& t,
               const std::vector<T1>& x,
               const std::vector<T2>& parms,
               const std::vector<double>& rate,
               const std::vector<int>& dummy,
               std::ostream* pstream__) const {
      typedef typename stan::return_type<T0, T1, T2>::type scalar;
      
      size_t n = x.size();
      std::vector<scalar> res(n);
      Matrix<scalar, Dynamic, Dynamic> m(n, n);
      Matrix<scalar, Dynamic, 1> v(n);
      for (size_t i = 0; i < n * n; ++i) m(i) = parms[i];
      for (size_t i = 0; i < n; ++i) v(i) = x[i];
      torsten::PMXLin<scalar> rv = m * v;
      for (size_t i = 0; i < n; ++i) res[i] = rv(i);

      return res;
    }
  };

  /**
   * Linear ODE model.
   *
   * @tparam T_time type of time
   * @tparam T_init type of init condition
   * @tparam T_rate type of dosing rate
   * @tparam T_par  type of parameters.
   */
  template<typename T_time, typename T_init, typename T_rate, typename T_par>
  class PMXLinODEModel {
    const Eigen::Matrix<T_par, -1, -1> & par_;
    const int ncmt_;

  public:
    static constexpr PMXLinODE f_ = PMXLinODE();
    using scalar_type = typename stan::return_type<T_time, T_init, T_rate, T_par>::type;
    using init_type   = T_init;
    using time_type   = T_time;
    using par_type    = T_par;
    using rate_type   = T_rate;

    PMXLinODEModel(const T_time& t0,
                   const PKRec<T_init>& y0,
                   const std::vector<T_rate> &rate,
                   const Eigen::Matrix<T_par, -1, -1>& par) :
      par_(par), ncmt_(y0.size())
    {}

    /*
     * return @c vars that will be steady-state
     * solution. For SS solution @c rate_ or @ y0_ will not
     * be in the solution.
     */
    template<typename T_a, typename T_r, typename T_ii>
    std::vector<stan::math::var> vars(const T_a& a, const T_r& r, const T_ii& ii) {
      return torsten::dsolve::pk_vars(a, r, ii, par_);
    }

    const Eigen::Matrix<T_par,-1,-1>  & par ()  const { return par_; }
    const PMXLinODE            & f()     const { return f_; }

    const int ncmt () const {
      return ncmt_;
    }

    template<typename Tt0, typename Tt1, typename T, typename T1>
    void solve(PKRec<T>& y,
               const Tt0& t0, const Tt1& t1,
               const std::vector<T1>& rate,
               const PMXOdeIntegrator<Analytical>& integ) const {
      using stan::math::matrix_exp;
      using stan::math::mdivide_left;
      using stan::math::multiply;

      typename stan::return_type_t<Tt0, Tt1> dt = t1 - t0;

      const int nCmt = par_.cols();
      Eigen::Matrix<scalar_type, -1, 1> y0t(nCmt);
      for (int i = 0; i < nCmt; ++i) y0t(i) = y(i);

      Eigen::Matrix<T, -1, 1> pred(nCmt);

      if (std::any_of(rate.begin(), rate.end(),
                      [](T_rate r){return r != 0;})) {
        Eigen::Matrix<T, -1, 1> rate_vec(rate.size()), x(nCmt), x2(nCmt);
        for (size_t i = 0; i < rate.size(); ++i) rate_vec(i) = rate[i];
        x = mdivide_left(par_, rate_vec);
        x2 = x + y;
        Eigen::Matrix<T, -1, -1> dt_system = multiply(dt, par_);
        pred = matrix_exp(dt_system) * x2;
        pred -= x;
      } else {
        Eigen::Matrix<T, -1, -1> dt_system = multiply(dt, par_);
        pred = matrix_exp(dt_system) * y0t;
      }
      y = pred;
    }

    template<typename Tt0, typename Tt1, typename T, typename T1>
    void solve(PKRec<T>& y,
               const Tt0& t0, const Tt1& t1,
               const std::vector<T1>& rate) const {
      const PMXOdeIntegrator<Analytical> integ;
      solve(y, t0, t1, rate, integ);
    }

    /*
     * solve the linear ODE: steady state version
     */
    template<typename T_amt, typename T_r, typename T_ii>
    Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> 
    solve(double t0, const T_amt& amt, const T_r& rate, const T_ii& ii,
          const int& cmt) const {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::matrix_exp;
      using stan::math::mdivide_left;
      using stan::math::multiply;
      using std::vector;

      typedef typename promote_args<T_ii, T_par>::type T0;
      typedef typename promote_args<T_amt, T_r, T_ii, T_par>::type scalar; //NOLINT

      int nCmt = ncmt();
      Matrix<T0, Dynamic, Dynamic> workMatrix;
      Matrix<T0, Dynamic, Dynamic> ii_system = multiply(ii, par_);
      Matrix<scalar, 1, Dynamic> pred(nCmt);
      pred.setZero();
      Matrix<scalar, Dynamic, 1> amounts(nCmt);
      amounts.setZero();

      if (rate == 0) {  // bolus dose
        amounts(cmt - 1) = amt;
        workMatrix = - matrix_exp(ii_system);
        for (int i = 0; i < nCmt; i++) workMatrix(i, i) += 1;
        amounts = mdivide_left(workMatrix, amounts);
        pred = multiply(matrix_exp(ii_system), amounts);

      } else if (ii > 0) {  // multiple truncated infusions
        scalar delta = amt / rate;
        static const char* function("Steady State Event");
        torsten::check_mti(amt, delta, ii, function);

        amounts(cmt - 1) = rate;
        scalar t = delta;
        amounts = mdivide_left(par_, amounts);
        Matrix<scalar, Dynamic, Dynamic> t_system = multiply(delta, par_);
        pred = matrix_exp(t_system) * amounts;
        pred -= amounts;

        workMatrix = - matrix_exp(ii_system);
        for (int i = 0; i < nCmt; i++) workMatrix(i, i) += 1;

        Matrix<scalar, Dynamic, 1> pred_t = pred.transpose();
        pred_t = mdivide_left(workMatrix, pred_t);
        t = ii - t;
        t_system = multiply(t, par_);
        pred_t = matrix_exp(t_system) * pred_t;
        pred = pred_t.transpose();

      } else {  // constant infusion
        amounts(cmt - 1) -= rate;
        pred = mdivide_left(par_, amounts);
      }
      return pred;
    }

    /*
     * wrapper to fit @c PrepWrapper's call signature
     */
    template<typename T_amt, typename T_r, typename T_ii>
    Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
    solve(double t0, const T_amt& amt, const T_r& rate, const T_ii& ii, const int& cmt,
          const PMXOdeIntegrator<torsten::Analytical>& integrator) const {
      return solve(t0, amt, rate, ii, cmt);
    }

  };

  template<typename T_time, typename T_init, typename T_rate, typename T_par>
  constexpr PMXLinODE PMXLinODEModel<T_time, T_init, T_rate, T_par>::f_;

}

#endif
