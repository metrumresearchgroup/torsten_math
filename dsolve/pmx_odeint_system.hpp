#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_ODEINT_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_ODEINT_SYSTEM_HPP

#include <stan/math/torsten/gradient_solution.hpp>
#include <stan/math/torsten/dsolve/cvodes_service.hpp>
#include <stan/math/torsten/dsolve/ode_forms.hpp>
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

  template <typename F, typename Tt, typename T_init, typename T_par>
  struct PMXOdeintSystem {
    using Ode = PMXOdeintSystem<F, Tt, T_init, T_par>;
    using scalar_t = typename torsten::return_t<T_init, T_par>::type;
    static constexpr bool is_var_y0  = stan::is_var<T_init>::value;
    static constexpr bool is_var_par = stan::is_var<T_par>::value;

    const F& f_;
    const double t0_;
    const std::vector<double>& ts_;
    const std::vector<T_init>& y0_;
    const std::vector<T_par>& theta_;
    const std::vector<double>& x_r_;
    const std::vector<int>& x_i_;
    const size_t N_;
    const size_t M_;
    const size_t ns;
    const size_t size_;
    std::ostream* msgs_;
    std::vector<double>& y0_fwd_system;
  private:
    int step_counter_;  

  public:
    template<typename ode_t>
    PMXOdeintSystem(dsolve::PMXOdeService<ode_t>& serv,
                    const F& f,
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
        ns(serv.ns),
        size_(serv.size),
        msgs_(msgs),
        y0_fwd_system(serv.y),
        step_counter_(0)
    {
      // initial state
      if (is_var_y0)  {
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
      rhs_impl(y, dy_dt, t, y0_, theta_);
      stan::math::check_size_match("PMXOdeintSystem", "y", y.size(), "dy_dt", dy_dt.size());
    }

    void rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                  const std::vector<double>& y0,
                  const std::vector<double>& theta) const;

    void rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                  const std::vector<double>& y0,
                  const std::vector<stan::math::var>& theta) const;

    void rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                  const std::vector<stan::math::var>& y0,
                  const std::vector<double>& theta) const;

    void rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                  const std::vector<stan::math::var>& y0,
                  const std::vector<stan::math::var>& theta) const;
  };

  /*
   * Data-only version
   */
  template<typename F, typename Tt, typename T_init, typename T_par>
  void PMXOdeintSystem<F, Tt, T_init, T_par>::rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                   const std::vector<double>& y0,
                                                   const std::vector<double>& theta) const
  {
    dy_dt = f_(t, y, theta_, x_r_, x_i_, msgs_);
  }

  /*
   * data @c y0, parameter @c theta
   */
  template<typename F, typename Tt, typename T_init, typename T_par>
  void PMXOdeintSystem<F, Tt, T_init, T_par>::rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                   const std::vector<double>& y0,
                                                   const std::vector<stan::math::var>& theta) const
  {
    using std::vector;
    using stan::math::var;

    try {
      stan::math::start_nested();

      vector<stan::math::var> y_vars(y.begin(), y.begin() + N_);
      vector<stan::math::var> theta_vars(M_);
      std::transform(theta.begin(), theta.end(), theta_vars.begin(),
                     [](const T_par& v) {return v.val();});

      vector<stan::math::var> dy_dt_vars = f_(t, y_vars, theta_vars, x_r_, x_i_, msgs_);

      stan::math::check_size_match("PMXOdeintSystem", "dz_dt", dy_dt_vars.size(), "states", N_);

      GradientSolution::fill_fwd_sens_ode_rhs<T_init, T_par>(dy_dt, y, dy_dt_vars, y_vars, theta_vars);
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();
  }

  /*
   * parameter @c y0, data @c theta
   */
  template<typename F, typename Tt, typename T_init, typename T_par>
  void PMXOdeintSystem<F, Tt, T_init, T_par>::rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                            const std::vector<stan::math::var>& y0,
                                                            const std::vector<double>& theta) const
  {
    using std::vector;

    try {
      stan::math::start_nested();

      vector<stan::math::var> y_vars(y.begin(), y.begin() + N_);
      vector<stan::math::var> dy_dt_vars = f_(t, y_vars, theta, x_r_, x_i_, msgs_);

      stan::math::check_size_match("PMXOdeintSystem", "dz_dt", dy_dt_vars.size(), "states", N_);

      GradientSolution::fill_fwd_sens_ode_rhs<T_init, T_par>(dy_dt, y, dy_dt_vars, y_vars, theta);
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();
  }

  /*
   * all parameter
   */
  template<typename F, typename Tt, typename T_init, typename T_par>
  void PMXOdeintSystem<F, Tt, T_init, T_par>::rhs_impl(const std::vector<double>& y, std::vector<double>& dy_dt, double t,
                                                            const std::vector<stan::math::var>& y0,
                                                            const std::vector<stan::math::var>& theta) const
  {
    using std::vector;

    try {
      stan::math::start_nested();

      vector<stan::math::var> y_vars(y.begin(), y.begin() + N_);
      vector<stan::math::var> theta_vars(M_);
      std::transform(theta.begin(), theta.end(), theta_vars.begin(),
                     [](const T_par& v) {return v.val();});

      vector<stan::math::var> dy_dt_vars = f_(t, y_vars, theta_vars, x_r_, x_i_, msgs_);

      stan::math::check_size_match("PMXOdeintSystem", "dz_dt", dy_dt_vars.size(), "states", N_);

      GradientSolution::fill_fwd_sens_ode_rhs<T_init, T_par>(dy_dt, y, dy_dt_vars, y_vars, theta_vars);
    } catch (const std::exception& e) {
      stan::math::recover_memory_nested();
      throw;
    }
    stan::math::recover_memory_nested();    
  }

}  // namespace dsolve
}  // namespace torsten
#endif
