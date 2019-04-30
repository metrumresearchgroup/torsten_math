#ifndef STAN_MATH_TORSTEN_FWD_SENS_SOLUTION_HPP
#define STAN_MATH_TORSTEN_FWD_SENS_SOLUTION_HPP

#include <stan/math/rev/core/precomputed_gradients.hpp>

namespace torsten {
  /*
   * Assuming ODE fwd sensitivity solution is aligned as
   *
   * (y1, y2, y3, ..., dy1/dp1, dy2/dp1, dy3/dp1, ... dy1/dpn, dy2/dpn...)
   */
  struct GradientSolution
  {
    static inline std::vector<stan::math::var>
    to_var(const std::vector<double>& y, const std::vector<stan::math::var>& theta)
    {
      const size_t n = y.size()/(1 + theta.size());
      std::vector<stan::math::var> y_res(n);
      std::vector<double> g(theta.size());
      for (size_t j = 0; j < n; j++) {
        for (size_t k = 0; k < theta.size(); k++) g[k] = y[n + n * k + j];
        y_res[j] = precomputed_gradients(y[j], theta, g);
      }
      return y_res;
    }

    /*
     * contribution of df/dy and df/dp to dy/dp in y' = f(y, p, t);
     */
    template<typename Ty, typename Tp>
    static inline double adj_of(const std::vector<Tp>& v, size_t n, size_t j);

    template<typename Ty, typename Tp>
    static inline void
    fill_fwd_sens_ode_rhs(std::vector<double>& dy_dt,
                          const std::vector<double>& y,
                          std::vector<stan::math::var>& rhs,
                          const std::vector<stan::math::var>& yv, const std::vector<Tp>& pv) {
      {
        const size_t n = yv.size();
        const size_t m = pv.size();
        const size_t ns = (stan::is_var<Ty>::value ? n : 0) + (stan::is_var<Tp>::value ? m : 0);
        for (size_t i = 0; i < n; i++) {
          dy_dt[i] = rhs[i].val();
          stan::math::set_zero_all_adjoints_nested();
          rhs[i].grad();
          for (size_t j = 0; j < ns; j++) {
            double g = adj_of<Ty, Tp>(pv, n, j);
            for (size_t k = 0; k < n; k++) g += y[n + n * j + k] * yv[k].adj();
            dy_dt[n + n * j + i] = g;
          }
        }
      }      
    }
  };

  template<>
  inline double GradientSolution::adj_of<stan::math::var, double>(const std::vector<double>& v, size_t n, size_t j)
  {
    return 0.0;
  }

  template<>
  inline double GradientSolution::adj_of<double, stan::math::var>(const std::vector<stan::math::var>& v, size_t n, size_t j)
  {
    return v[j].adj();
  }

  template<>
  inline double GradientSolution::adj_of<stan::math::var, stan::math::var>(const std::vector<stan::math::var>& v, size_t n, size_t j)
  {
    return (j < n ? 0 : v[j - n].adj());
  }

}

#endif
