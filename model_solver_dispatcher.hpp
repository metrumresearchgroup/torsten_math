#ifndef STAN_MATH_TORSTEN_MODEL_SOLVER_DISPATCHER_HPP
#define STAN_MATH_TORSTEN_MODEL_SOLVER_DISPATCHER_HPP

namespace torsten {

  /*
   * Dispatcher a model's solver according to the model type,
   * return @c var or data, and is stead state or not.
   */
  template<bool GenVar, typename T_model, typename... T_pred>
  struct ModelSolverDispatcher;

  /*
   * General version of the dispatcher.
   */
  template<typename T_model, typename... T_pred>
  struct ModelSolverDispatcher<true, T_model, T_pred...> {
    template<typename T_time>
    static auto solve(const T_model& pkmodel, const T_time& dt, T_pred const&... pred_pars) { //NOLINT
      return pkmodel.solve(dt, pred_pars...);
    }

    template<typename T_amt, typename T_r, typename T_ii>
    static auto solve(const T_model& pkmodel, const T_amt& amt, const T_r& rate, const T_ii& ii, const int& cmt, T_pred const&... pred_pars) { // NOLINT
      return pkmodel.solve(amt, rate, ii, cmt, pred_pars...);
    }
  };

  /*
   * For most built-in models, we have for template
   * parameters for model constructor: @c T_time, @c T_init,
   * @c T_rate, @c T_par. When all are data, the dispatcher
   * simply passed it on to the model's solver.
   */
  template<template<typename...> class T_model, typename... T_pred>
  struct ModelSolverDispatcher<false, T_model<double, double, double, double>, T_pred...> { // NOLITN

    using model_t = T_model<double, double, double, double>;

    static auto solve(const model_t& pkmodel, const double& dt, T_pred const&... pred_pars) { //NOLINT
      return pkmodel.solve(dt, pred_pars...);
    }

    static auto solve(const model_t& pkmodel, const double& amt, const double& rate, const double& ii, const int& cmt, T_pred const&... pred_pars) { // NOLINT
      return pkmodel.solve(amt, rate, ii, cmt, pred_pars...);
    }
  };

  /*
   * Sometimes we need return solution in form of data. An
   * example would be in MPI version, where we need pass
   * data through MPI before generating @c var results.
   */
  template<template<typename...> class T_model,
           typename T_time, typename T_init, typename T_rate, typename T_par, typename... T_pred> // NOLINT
  struct ModelSolverDispatcher<false, T_model<T_time, T_init, T_rate, T_par>, T_pred...> { // NOLITN
    using model_t = T_model<T_time, T_init, T_rate, T_par>;

    static Eigen::VectorXd solve(const model_t& pkmodel, const T_time& dt, T_pred const&... pred_pars) { //NOLINT
      using std::vector;
      using Eigen::VectorXd;
      using Eigen::Matrix;
      using stan::math::value_of;
      using stan::math::var;

      const Matrix<T_init, 1, -1>& y0 = pkmodel.y0();
      const vector<T_rate>& rate = pkmodel.rate();
      const vector<T_par>& par = pkmodel.par();      

      VectorXd res_d;

      stan::math::start_nested();

      Matrix<T_init, 1, -1> y0_new(y0.size());
      vector<T_rate> rate_new(rate.size());
      vector<T_par> par_new(par.size());

      for (int i = 0; i < y0_new.size(); ++i) {y0_new(i) = value_of(y0(i));}
      for (size_t i = 0; i < rate_new.size(); ++i) {rate_new[i] = value_of(rate[i]);}
      for (size_t i = 0; i < par_new.size(); ++i) {par_new[i] = value_of(par[i]);}

      T_time t0 = value_of(pkmodel.t0());
      T_time t1 = value_of(pkmodel.t0()) + value_of(dt);
      T_time dt_new = t1 - t0;
      model_t pkmodel_new(t0, y0_new, rate_new, par_new);

      auto res = pkmodel_new.solve(dt_new, pred_pars...);
      vector<var> var_new(pkmodel_new.vars(t1));
      vector<double> g;
      const int nx = res.size();
      const int ny = var_new.size();      
      res_d.resize(nx * (ny + 1));
      for (int i = 0; i < nx; ++i) {
        stan::math::set_zero_all_adjoints_nested();
        res_d(i * (ny + 1)) = res[i].val();
        res[i].grad(ny, g);
        for (int j = 0; j < ny; ++j) {
          res_d(i * (ny + 1) + j + 1) = g[j];
        }
      }

      return res_d;
    }

    template<typename T_amt, typename T_r, typename T_ii>
    static Eigen::VectorXd solve(const model_t& pkmodel, const T_amt& amt, const T_r& r, const T_ii& ii, const int& cmt, T_pred const&... pred_pars) { // NOLINT
      using std::vector;
      using Eigen::VectorXd;
      using Eigen::Matrix;
      using stan::math::value_of;
      using stan::math::var;

      const Matrix<T_init, 1, -1>& y0 = pkmodel.y0();
      const vector<T_rate>& rate = pkmodel.rate();
      const vector<T_par>& par = pkmodel.par();      

      VectorXd res_d;

      stan::math::start_nested();

      Matrix<T_init, 1, -1> y0_new(y0.size());
      vector<T_rate> rate_new(rate.size());
      vector<T_par> par_new(par.size());

      for (int i = 0; i < y0_new.size(); ++i) {y0_new(i) = value_of(y0(i));}
      for (size_t i = 0; i < rate_new.size(); ++i) {rate_new[i] = value_of(rate[i]);}
      for (size_t i = 0; i < par_new.size(); ++i) {par_new[i] = value_of(par[i]);}

      T_time t0 = value_of(pkmodel.t0());
      model_t pkmodel_new(t0, y0_new, rate_new, par_new);

      T_amt amt_new = value_of(amt);
      T_r r_new = value_of(r);
      T_ii ii_new = value_of(ii);
      auto res = pkmodel_new.solve(amt_new, r_new, ii_new, cmt, pred_pars...);
      vector<var> var_new(pkmodel_new.vars(amt_new, r_new, ii_new));
      vector<double> g;
      const int nx = res.size();
      const int ny = var_new.size();      
      res_d.resize(nx * (ny + 1));
      for (int i = 0; i < nx; ++i) {
        stan::math::set_zero_all_adjoints_nested();
        res_d(i * (ny + 1)) = res[i].val();
        res[i].grad(ny, g);
        for (int j = 0; j < ny; ++j) {
          res_d(i * (ny + 1) + j + 1) = g[j];
        }
      }

      return res_d;
    }
  };


}

#endif
