#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_ODE_VARS_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_ODE_VARS_HPP

#include <stan/math/torsten/dsolve/pk_vars.hpp>

namespace torsten {
  namespace dsolve {

    inline const std::vector<stan::math::var>&
    pmx_ode_vars(const std::vector<stan::math::var>& y0, 
                 const std::vector<double>& theta,
                 const std::vector<double>& ts) {
      return y0;
    }

    inline const std::vector<stan::math::var>&
    pmx_ode_vars(const std::vector<double>& y0, 
                 const std::vector<stan::math::var>& theta,
                 const std::vector<double>& ts) {
      return theta;
    }

    inline const std::vector<stan::math::var>&
    pmx_ode_vars(const std::vector<double>& y0, 
                 const std::vector<double>& theta,
                 const std::vector<stan::math::var>& ts) {
      return ts;
    }

    /**
     * General ODE system that contains informtion on residual
     * equation functor, sensitivity residual equation functor,
     * as well as initial conditions. This is a base type that
     * is intended to contain common values used by forward
     * sensitivity system.
     *
     * @tparam Tt scalar type of time steps
     * @tparam Ty scalar type of initial unknown values
     * @tparam Tp scalar type of parameters
     */
    template <typename Tt, typename Ty, typename Tp>
    inline const std::vector<stan::math::var>
    pmx_ode_vars(const std::vector<Ty>& y0, 
                 const std::vector<Tp>& theta,
                 const std::vector<Tt>& ts) {
      return pk_vars(y0, theta, ts);
    }
  }
}

#endif
