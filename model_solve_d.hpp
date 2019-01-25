#ifndef STAN_MATH_TORSTEN_MODEL_SOLVE_D_HPP
#define STAN_MATH_TORSTEN_MODEL_SOLVE_D_HPP

namespace torsten {
  template<typename T_model,
           typename std::enable_if_t<!stan::is_var<typename T_model::scalar_type>::value >* = nullptr> //NOLINT
  Eigen::VectorXd model_solve_d(const T_model& pkmodel, const double& dt) {
    return pkmodel.solve(dt);
  }

  template<typename T_model>
  Eigen::VectorXd model_solve_d(const T_model& pkmodel,
                                typename T_model::time_type const& dt) {
    using std::vector;
    using Eigen::VectorXd;
    using Eigen::Matrix;
    using stan::math::value_of;
    using stan::math::var;

    using T_time = typename T_model::time_type; 
    using T_init = typename T_model::init_type;
    using T_rate = typename T_model::rate_type;
    using T_par = typename T_model::par_type;

    const Matrix<T_init, 1, -1>& y0 = pkmodel.y0();
    const vector<T_rate>& rate = pkmodel.rate();
    const vector<T_par>& par = pkmodel.par();      

    VectorXd res_d;

    stan::math::start_nested();

    try {
      Matrix<T_init, 1, -1> y0_new(y0.size());
      vector<T_rate> rate_new(rate.size());
      vector<T_par> par_new(par.size());

      for (int i = 0; i < y0_new.size(); ++i) {y0_new(i) = value_of(y0(i));}
      for (size_t i = 0; i < rate_new.size(); ++i) {rate_new[i] = value_of(rate[i]);}
      for (size_t i = 0; i < par_new.size(); ++i) {par_new[i] = value_of(par[i]);}

      T_time t0 = value_of(pkmodel.t0());
      T_time t1 = value_of(pkmodel.t0()) + value_of(dt);
      T_time dt_new = t1 - t0;
      T_model pkmodel_new(t0, y0_new, rate_new, par_new);

      auto res = pkmodel_new.solve(dt_new);
      vector<var> var_new(pkmodel_new.vars(t1));
      vector<double> g;
      const int nx = res.size();
      const int ny = var_new.size();      
      res_d.resize(nx * (ny + 1));
      for (int i = 0; i < nx; ++i) {
        stan::math::set_zero_all_adjoints_nested();
        res_d(i * (ny + 1)) = res[i].val();
        res[i].grad(var_new, g);
        for (int j = 0; j < ny; ++j) {
          res_d(i * (ny + 1) + j + 1) = g[j];
        }
      }
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();

    return res_d;
  }

  template<typename T_model,
           typename std::enable_if_t<!stan::is_var<typename T_model::par_type>::value >* = nullptr>
    Eigen::VectorXd model_solve_d(const T_model& pkmodel, const double& amt, const double& rate, const double& ii, const int& cmt) { // NOLINT
      return pkmodel.solve(amt, rate, ii, cmt);
  }

  template<typename T_model, typename T_amt, typename T_rate, typename T_ii>
  Eigen::VectorXd model_solve_d(const T_model& pkmodel, const T_amt& amt, const T_rate& r, const T_ii& ii, const int& cmt) { // NOLINT
      using std::vector;
      using Eigen::VectorXd;
      using Eigen::Matrix;
      using stan::math::value_of;
      using stan::math::var;

      VectorXd res_d;

      using T_par = typename T_model::par_type;
      const std::vector<T_par>& par(pkmodel.par());

      stan::math::start_nested();
      try {
        std::vector<T_par> par_new(par.size());
        for (size_t i = 0; i < par.size(); ++i) par_new[i] = value_of(par[i]);

        T_model pkmodel_new(pkmodel.t0(), pkmodel.y0(), pkmodel.rate(), par_new);

        T_amt amt_new = value_of(amt);
        T_rate r_new = value_of(r);
        T_ii ii_new = value_of(ii);
        auto res = pkmodel_new.solve(amt_new, r_new, ii_new, cmt);
        vector<var> var_new(pkmodel_new.vars(amt_new, r_new, ii_new));
        vector<double> g;
        const int nx = res.size();
        const int ny = var_new.size();      
        res_d.resize(nx * (ny + 1));
        for (int i = 0; i < nx; ++i) {
          stan::math::set_zero_all_adjoints_nested();
          res_d(i * (ny + 1)) = res[i].val();
          res[i].grad(var_new, g);
          for (int j = 0; j < ny; ++j) {
            res_d(i * (ny + 1) + j + 1) = g[j];
          }
        }
      } catch (const std::exception& e) {
        stan::math::recover_memory_nested();
        throw;
      }
      stan::math::recover_memory_nested();

      return res_d;
    }

  
}
#endif
