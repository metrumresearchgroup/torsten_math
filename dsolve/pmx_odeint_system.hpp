#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_ODEINT_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_ODEINT_SYSTEM_HPP

#include <stan/math/torsten/return_type.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/fun/value_of.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace torsten {
namespace dsolve {

  template <typename F, typename T_init, typename T_par>
  struct PMXOdeintSystem {
    static constexpr bool is_y0_var  = stan::is_var<T_init>::value;
    static constexpr bool is_par_var = stan::is_var<T_par>::value;

    const F& f_;
    const double t0_;
    const std::vector<double>& ts_;
    const std::vector<T_init>& y0_;
    const std::vector<T_par>& theta_;
    const std::vector<double>& x_r_;
    const std::vector<int>& x_i_;
    const size_t N_;
    const size_t M_;
    const size_t size_;
    std::vector<std::vector<double> > y_res_;
    std::ostream* msgs_;
    std::vector<double> y0_fwd_system;
  private:
    int step_counter_;  

  public:
    PMXOdeintSystem(const F& f,
                    double t0,
                    const std::vector<double>& ts,
                    const std::vector<T_init>& y0,
                    const std::vector<T_par>& theta,
                    const std::vector<double>& x_r,
                    const std::vector<int>& x_i,
                    std::ostream* msgs)
      : f_(f),
        t0_(t0),
        ts_(ts),
        y0_(y0),
        theta_(theta),
        x_r_(x_r),
        x_i_(x_i),
        N_(y0.size()),
        M_(theta.size()),      
        size_(N_ + N_ * ((is_y0_var ? N_ : 0) + (is_par_var ? M_ : 0))),
        y_res_(ts_.size(), std::vector<double>(size_, 0.0)),
        msgs_(msgs),
        y0_fwd_system(size_, 0.0),
        step_counter_(0)
    {
      // initial state
      if (is_y0_var)  {
      std::transform(y0.begin(), y0.end(), y0_fwd_system.begin(),
                     [](const T_init& v){ return stan::math::value_of(v); });        
      for (size_t i = 0; i < N_; i++) y0_fwd_system[N_ + i * N_ + i] = 1.0;
      } else {
      std::transform(y0.begin(), y0.end(), y0_fwd_system.begin(),
                     [](const T_init& v){ return stan::math::value_of(v); });
      }
    }

    void operator()(const std::vector<double>& y, std::vector<double>& dy_dt,
                    double t) const {
      eval_fwd_sens_rhs(y, dy_dt, t, y0_, theta_);
      stan::math::check_size_match("PMXOdeintSystem", "y", y.size(), "dy_dt", dy_dt.size());
    }

    /*
     * For getting results as an @c Odeint @c observer. We
     * don't need to saves results at @c t0.
     */
    void operator()(const std::vector<double>& curr_result, double t) {
      if(t > t0_) {
        y_res_[step_counter_] = curr_result;
        step_counter_++;
      }
    }

    void eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                           const std::vector<double>& y0,
                           const std::vector<double>& theta) const;

    void eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                           const std::vector<double>& y0,
                           const std::vector<stan::math::var>& theta) const;

    void eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                           const std::vector<stan::math::var>& y0,
                           const std::vector<double>& theta) const;

    void eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                           const std::vector<stan::math::var>& y0,
                           const std::vector<stan::math::var>& theta) const;

    std::vector<std::vector<typename torsten::return_t<T_init, T_par>::type >>
    decouple_states(const std::vector<std::vector<double>>& y) const
    {
      return decouple_states_impl(y, y0_, theta_);
    }

    std::vector<std::vector<typename torsten::return_t<T_init, T_par>::type >>
    decouple_states_impl(const std::vector<std::vector<double>>& y,
                         const std::vector<double>& y0,
                         const std::vector<double>& theta) const
    {
      return y;
    }

    std::vector<std::vector<typename torsten::return_t<T_init, T_par>::type >>
    decouple_states_impl(const std::vector<std::vector<double>>& y,
                         const std::vector<double>& y0,
                         const std::vector<stan::math::var>& theta) const
    {
      std::vector<stan::math::var> temp_vars(N_);
      std::vector<double> temp_gradients(M_);
      std::vector<std::vector<stan::math::var> > y_return(y.size());

      for (size_t i = 0; i < y.size(); i++) {
        // iterate over number of equations
        for (size_t j = 0; j < N_; j++) {
          // iterate over parameters for each equation
          for (size_t k = 0; k < M_; k++)
            temp_gradients[k] = y[i][N_ + N_ * k + j];

          temp_vars[j] = precomputed_gradients(y[i][j], theta_, temp_gradients);
        }
        y_return[i] = temp_vars;
      }
      return y_return;      
    }

    std::vector<std::vector<typename torsten::return_t<T_init, T_par>::type >>
    decouple_states_impl(const std::vector<std::vector<double>>& y,
                         const std::vector<stan::math::var>& y0,
                         const std::vector<double>& theta) const
    {
      using std::vector;

      vector<stan::math::var> temp_vars(N_);
      vector<double> temp_gradients(N_);
      vector<vector<stan::math::var> > y_return(y.size());

      for (size_t i = 0; i < y.size(); i++) {
        // iterate over number of equations
        for (size_t j = 0; j < N_; j++) {
          // iterate over parameters for each equation
          for (size_t k = 0; k < N_; k++)
            temp_gradients[k] = y[i][N_ + N_ * k + j];

          temp_vars[j] = precomputed_gradients(y[i][j], y0_, temp_gradients);
        }
        y_return[i] = temp_vars;
      }
      return y_return;
    }

    std::vector<std::vector<typename torsten::return_t<T_init, T_par>::type >>
    decouple_states_impl(const std::vector<std::vector<double>>& y,
                         const std::vector<stan::math::var>& y0,
                         const std::vector<stan::math::var>& theta) const
    {
      using std::vector;

      vector<stan::math::var> vars = y0_;
      vars.insert(vars.end(), theta_.begin(), theta_.end());

      vector<stan::math::var> temp_vars(N_);
      vector<double> temp_gradients(N_ + M_);
      vector<vector<stan::math::var> > y_return(y.size());

      for (size_t i = 0; i < y.size(); i++) {
        // iterate over number of equations
        for (size_t j = 0; j < N_; j++) {
          // iterate over parameters for each equation
          for (size_t k = 0; k < N_ + M_; k++)
            temp_gradients[k] = y[i][N_ + N_ * k + j];

          temp_vars[j] = precomputed_gradients(y[i][j], vars, temp_gradients);
        }
        y_return[i] = temp_vars;
      }

      return y_return;
    }

  };

  /*
   * Data-only version
   */
  template<typename F, typename T_init, typename T_par>
  void PMXOdeintSystem<F, T_init, T_par>::eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                            const std::vector<double>& y0,
                                                            const std::vector<double>& theta) const
  {
    dy_dt = f_(t, y, theta_, x_r_, x_i_, msgs_);
  }

  /*
   * data @c y0, parameter @c theta
   */
  template<typename F, typename T_init, typename T_par>
  void PMXOdeintSystem<F, T_init, T_par>::eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                            const std::vector<double>& y0,
                                                            const std::vector<stan::math::var>& theta) const
  {
    using std::vector;
    using stan::math::var;

    vector<double> grad(N_ + M_);

    try {
      stan::math::start_nested();

      vector<stan::math::var> z_vars;
      z_vars.reserve(N_ + M_);

      vector<stan::math::var> y_vars(y.begin(), y.begin() + N_);
      z_vars.insert(z_vars.end(), y_vars.begin(), y_vars.end());

      // vector<stan::math::var> theta_vars(theta_dbl_.begin(), theta_dbl_.end());
      vector<stan::math::var> theta_vars(M_);
      std::transform(theta.begin(), theta.end(), theta_vars.begin(),
                     [](const T_par& v) {return v.val();});
      z_vars.insert(z_vars.end(), theta_vars.begin(), theta_vars.end());

      vector<stan::math::var> dy_dt_vars = f_(t, y_vars, theta_vars, x_r_, x_i_, msgs_);

      stan::math::check_size_match("PMXOdeintSystem", "dz_dt", dy_dt_vars.size(), "states", N_);

      for (size_t i = 0; i < N_; i++) {
        dy_dt[i] = dy_dt_vars[i].val();
        stan::math::set_zero_all_adjoints_nested();
        dy_dt_vars[i].grad(z_vars, grad);

        for (size_t j = 0; j < M_; j++) {
          // orders derivatives by equation (i.e. if there are 2 eqns
          // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as:
          // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
          double temp_deriv = grad[N_ + j];
          for (size_t k = 0; k < N_; k++)
            temp_deriv += y[N_ + N_ * j + k] * grad[k];

          dy_dt[N_ + i + j * N_] = temp_deriv;
        }
      }
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();
  }

  /*
   * parameter @c y0, data @c theta
   */
  template<typename F, typename T_init, typename T_par>
  void PMXOdeintSystem<F, T_init, T_par>::eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                            const std::vector<stan::math::var>& y0,
                                                            const std::vector<double>& theta) const
  {
    using std::vector;

    std::vector<double> grad(N_);

    try {
      stan::math::start_nested();

      vector<stan::math::var> y_vars(y.begin(), y.begin() + N_);
      vector<stan::math::var> dy_dt_vars = f_(t, y_vars, theta, x_r_, x_i_, msgs_);

      stan::math::check_size_match("PMXOdeintSystem", "dz_dt", dy_dt_vars.size(), "states", N_);

      for (size_t i = 0; i < N_; i++) {
        dy_dt[i] = dy_dt_vars[i].val();
        stan::math::set_zero_all_adjoints_nested();
        dy_dt_vars[i].grad(y_vars, grad);

        for (size_t j = 0; j < N_; j++) {
          // orders derivatives by equation (i.e. if there are 2 eqns
          // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as:
          // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
          double temp_deriv = 0;
          for (size_t k = 0; k < N_; k++)
            temp_deriv += y[N_ + N_ * j + k] * grad[k];

          dy_dt[N_ + i + j * N_] = temp_deriv;
        }
      }
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();
  }

  /*
   * all parameter
   */
  template<typename F, typename T_init, typename T_par>
  void PMXOdeintSystem<F, T_init, T_par>::eval_fwd_sens_rhs(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                            const std::vector<stan::math::var>& y0,
                                                            const std::vector<stan::math::var>& theta) const
  {
    using std::vector;

    std::vector<double> grad(N_ + M_);

    try {
      stan::math::start_nested();

      vector<stan::math::var> z_vars;
      z_vars.reserve(N_ + M_);

      vector<stan::math::var> y_vars(y.begin(), y.begin() + N_);
      z_vars.insert(z_vars.end(), y_vars.begin(), y_vars.end());

      // vector<stan::math::var> theta_vars(theta_dbl_.begin(), theta_dbl_.end());
      vector<stan::math::var> theta_vars(M_);
      std::transform(theta.begin(), theta.end(), theta_vars.begin(),
                     [](const T_par& v) {return v.val();});
      z_vars.insert(z_vars.end(), theta_vars.begin(), theta_vars.end());

      vector<stan::math::var> dy_dt_vars = f_(t, y_vars, theta_vars, x_r_, x_i_, msgs_);

      stan::math::check_size_match("PMXOdeintSystem", "dz_dt", dy_dt_vars.size(), "states", N_);

      for (size_t i = 0; i < N_; i++) {
        dy_dt[i] = dy_dt_vars[i].val();
        stan::math::set_zero_all_adjoints_nested();
        dy_dt_vars[i].grad(z_vars, grad);

        for (size_t j = 0; j < N_ + M_; j++) {
          // orders derivatives by equation (i.e. if there are 2 eqns
          // (y1, y2) and 2 parameters (a, b), dy_dt will be ordered as:
          // dy1_dt, dy2_dt, dy1_da, dy2_da, dy1_db, dy2_db
          double temp_deriv = j < N_ ? 0 : grad[j];
          for (size_t k = 0; k < N_; k++)
            temp_deriv += y[N_ + N_ * j + k] * grad[k];

          dy_dt[N_ + i + j * N_] = temp_deriv;
        }
      }
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();    
  }

}  // namespace dsolve
}  // namespace torsten
#endif
