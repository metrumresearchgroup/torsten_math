#ifndef STAN_MATH_TORSTEN_MODEL_SOLVE_D_HPP
#define STAN_MATH_TORSTEN_MODEL_SOLVE_D_HPP

#include <stan/math/torsten/torsten_def.hpp>
#include <stan/math/torsten/val_and_grad_nested.hpp>
#include <stan/math/torsten/pmx_ode_integrator.hpp>
#include <stan/math/torsten/dsolve/pk_vars.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <stan/math/torsten/val_and_grad_nested.hpp>

namespace torsten {

  template<typename model_t>
  struct pmx_model_vars {
    template<typename T0, typename T1, typename T2, typename T3>
    static std::vector<stan::math::var> vars(const T0& t1,
                                             const PKRec<T1>& y0,
                                             const std::vector<T2> &rate,
                                             const std::vector<T3> &par) {
      return torsten::dsolve::pk_vars(t1, y0, rate, par);
    }
  };

  template<typename... Ts>
  struct pmx_model_vars<PKODEModel<Ts...> > {
    template<typename T0, typename T1, typename T2, typename T3>
    static std::vector<stan::math::var> vars(const T0& t1,
                                             const PKRec<T1>& y0,
                                             const std::vector<T2> &rate,
                                             const std::vector<T3> &par) {
      return PKODEModel<Ts...>::vars(t1, y0, rate, par);
    }
  };

  /*
   * Solve PK models and return the results in form of data,
   * arrange as
   *
   *    sol value y1, dy1/dp1, dy1/dp2..., sol value y2, dy2/dp2, dy2/dp2...
   *
   * For transient solution, when the return type is already
   * data, we pass on to model's solver.
   */
  template<typename T_model, PMXOdeIntegratorId It,
           typename... Ts,
           typename std::enable_if_t<!stan::is_var<typename T_model::par_type>::value >* = nullptr> //NOLINT
  Eigen::VectorXd model_solve_d(const T_model& pkmodel,
                                const PKRec<double>& y,
                                const double& t0, const double& t1,
                                const std::vector<double>& rate,
                                const PMXOdeIntegrator<It>& integ,
                                const Ts... model_pars) {
    PKRec<double> yd(y);
    pkmodel.solve(yd, t0, t1, rate, integ);
    return yd;
  }

  /*
   * Solve PK models and return the results in form of data,
   * arrange as
   *
   *    sol value y1, dy1/dp1, dy1/dp2..., sol value y2, dy2/dp2, dy2/dp2...
   *
   * For transient solution. When the solution is @c var
   * type, we take gradient using autodiff.
   */
  template<typename T_model, typename T, 
           typename Tt0, typename Tt1,
           typename T1, PMXOdeIntegratorId It,
           typename... Ts,
           typename std::enable_if_t<It != torsten::PkBdf && It != torsten::PkAdams && It != torsten::PkRk45>* = nullptr>
  Eigen::VectorXd model_solve_d(const T_model& pkmodel,
                                const PKRec<T>& y,
                                const Tt0& t0, const Tt1& t1,
                                const std::vector<T1>& rate,
                                const PMXOdeIntegrator<It>& integ,
                                const Ts... model_pars) {
    using stan::math::value_of;
    using stan::math::var;

    Eigen::VectorXd res_d;
    stan::math::start_nested();
    try {
      PKRec<var> y_new(y.size());
      std::vector<T1> rate_new(rate.size());
      std::vector<typename T_model::par_type> par_new(pkmodel.par().size());

      for (int i = 0; i < y_new.size(); ++i) {y_new(i) = value_of(y(i));}
      for (size_t i = 0; i < rate_new.size(); ++i) {rate_new[i] = value_of(rate[i]);}
      for (size_t i = 0; i < par_new.size(); ++i) {par_new[i] = value_of(pkmodel.par()[i]);}

      Tt0 t0_ = value_of(t0);
      Tt1 t1_ = value_of(t1);
      T_model pkmodel_new(t0_, y, rate_new, par_new, model_pars...);

      std::vector<var> var_new = stan::is_var<T>::value ?
        pmx_model_vars<T_model>::vars(t1, y_new, rate_new, pkmodel_new.par()) :
        pmx_model_vars<T_model>::vars(t1, y, rate_new, pkmodel_new.par());
      pkmodel_new.solve(y_new, t0_, t1_, rate_new, integ);
      res_d = val_and_grad_nested(y_new, var_new);
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();

    return res_d;
  }

  template<typename T_model, typename T, 
           typename Tt0, typename Tt1,
           typename T1, PMXOdeIntegratorId It,
           typename... Ts,
           typename std::enable_if_t<It == torsten::PkBdf || It == torsten::PkAdams || It == torsten::PkRk45>* = nullptr>
  Eigen::VectorXd model_solve_d(const T_model& pkmodel,
                                const PKRec<T>& y,
                                const Tt0& t0, const Tt1& t1,
                                const std::vector<T1>& rate,
                                const PMXOdeIntegrator<It>& integ,
                                const Ts... model_pars) {
    Eigen::VectorXd yd;
    pkmodel.solve_d(yd, y, t0, t1, rate, integ);
    return yd;
  }

  /*
   * Solve PK models and return the results in form of data,
   * arrange as
   *
   *    sol value y1, dy1/dp1, dy1/dp2..., sol value y2, dy2/dp2, dy2/dp2...
   *
   * For steady-state solution. when the return type is already
   * data, we pass on to model's solver.
   */
  template<typename T_model, PMXOdeIntegratorId It, typename... Ts,
           typename std::enable_if_t<!stan::is_var<typename T_model::par_type>::value >* = nullptr> //NOLINT
    Eigen::VectorXd model_solve_d(const T_model& pkmodel,
                                  const double& amt, const double& rate, const double& ii, const int& cmt, // NOLINT
                                  const PMXOdeIntegrator<It>& integ,
                                  const Ts... model_pars) {
    return pkmodel.solve(amt, rate, ii, cmt, integ);
  }

  /*
   * Solve PK models and return the results in form of data,
   * arrange as
   *
   *    sol value y1, dy1/dp1, dy1/dp2..., sol value y2, dy2/dp2, dy2/dp2...
   *
   * For steady-state solution. When the solution is @c var
   * type, we take gradient using autodiff.
   */
  template<typename T_model, typename T_amt, typename T_rate, typename T_ii,
           PMXOdeIntegratorId It, typename... Ts>
  Eigen::VectorXd model_solve_d(const T_model& pkmodel,
                                const T_amt& amt, const T_rate& r, const T_ii& ii, const int& cmt, // NOLINT
                                const PMXOdeIntegrator<It>& integ,
                                const Ts... model_pars) {
    using std::vector;
    using Eigen::VectorXd;
    using Eigen::Matrix;
    using stan::math::value_of;
    using stan::math::var;

    VectorXd res_d;

    stan::math::start_nested();
    try {
      std::vector<typename T_model::par_type> par_new(pkmodel.par().size());
      for (size_t i = 0; i < par_new.size(); ++i) par_new[i] = value_of(pkmodel.par()[i]);

      T_model pkmodel_new(pkmodel.t0(), pkmodel.y0(), pkmodel.rate(), par_new, model_pars...);

      T_amt amt_new = value_of(amt);
      T_rate r_new = value_of(r);
      T_ii ii_new = value_of(ii);
      auto res = pkmodel_new.solve(amt_new, r_new, ii_new, cmt, integ);
      vector<var> var_new(pkmodel_new.vars(amt_new, r_new, ii_new));
      res_d = val_and_grad_nested(res, var_new);
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();

    return res_d;
  }


}
#endif
