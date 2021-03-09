#ifndef STAN_MATH_TORSTEN_DSOLVE_ODE_OBSERVER_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ODE_OBSERVER_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/torsten/mpi/precomputed_gradients.hpp>
#include <type_traits>

namespace torsten {
namespace dsolve {

  template<typename Ode>
  struct has_var_ts : std::true_type {};

  template<template<typename...> class ode_t, typename F, typename... Ts>
  struct has_var_ts<ode_t<F, double, Ts...>> : std::false_type {};

  template<typename Ode>
  struct has_var_y0 : std::true_type {};

  template<template<typename...> class ode_t, typename F, typename Tt, typename... Ts>
  struct has_var_y0<ode_t<F, Tt, double, Ts...>> : std::false_type {};

  template<typename Ode>
  struct has_var_par;

  template<template<typename...> class ode_t, typename F, typename Tt, typename T_init, typename... Ts>
  struct has_var_par<ode_t<F, Tt, T_init, Ts...>> : stan::is_var<stan::return_type_t<Ts...>> {};

  template<typename Ode>
  struct OdeObserver {
    using dbl_state_t = decltype(std::declval<Ode>().y0_fwd_system);
    using state_t = typename Ode::state_t;

    const Ode& ode;
    const size_t n;
    const size_t m;
    const size_t ns;
    std::vector<state_t> y;
    int step_counter_;
    double curr_t_;

    OdeObserver(const Ode& ode0) :
      ode(ode0), n(ode.N), m(ode.M), ns(ode.ns),
      y(ode.ts_.size(), Ode::null_state(n)),
      step_counter_(0)
    {}

    inline void reset() {
      step_counter_ = 0;
      for (size_t i = 0; i < ode.ts_.size(); ++i) {
        std::fill(y[i].begin(), y[i].end(), 0);        
      }
    }

    /**
     * use observer to convert y value and gradient to var
     * results, if necessary.
     */
    inline void operator()(const dbl_state_t& curr_result, double t) {
      curr_t_ = t;
      if(curr_t_ > ode.t0_) {
        observer_impl(y[step_counter_], curr_result);
        step_counter_++;
      }
    }

    /**
     * use observer to convert y value and gradient to var
     * results, if necessary.
     */
    inline void operator()(const N_Vector& curr_y, const N_Vector* curr_ys, double t) {
      curr_t_ = t;
      if(curr_t_ > ode.t0_) {
        observer_impl(y[step_counter_], curr_y, curr_ys);
        step_counter_++;
      }
    }

  private:

    /**
     * All data, return data
     */
    template<typename ode_type = Ode>
    inline stan::require_all_not_t<has_var_ts<ode_type>, has_var_y0<ode_type>, has_var_par<ode_type>>
    observer_impl(state_t& y_res, const dbl_state_t & y) const {
      std::copy(y.data(), y.data() + n, y_res.data());
    }

    /**
     * All data, return data
     */
    template<typename ode_type = Ode>
    inline stan::require_all_not_t<has_var_ts<ode_type>, has_var_y0<ode_type>, has_var_par<ode_type>>
    observer_impl(state_t& y_res, const N_Vector& y, const N_Vector* ys) const {
      std::copy(NV_DATA_S(y), NV_DATA_S(y) + n, y_res.data());
    }

    /**
     * When only @c ts is @c var, we don't solve
     * sensitivity ODE since the sensitivity is simply the RHS.
     */
    template<typename ode_type = Ode>
    inline std::enable_if_t<has_var_ts<ode_type>::value && (!has_var_y0<ode_type>::value) && (!has_var_par<ode_type>::value)>
    observer_impl(state_t& y_res, const dbl_state_t & y) const {
      auto g(ode_type::null_dbl_state(n * (1 + ode.ts_.size())));
      std::copy(y.data(), y.data() + n, g.data());
      dbl_state_t dydt = ode.dbl_rhs_impl(curr_t_, y);
      std::copy(dydt.data(), dydt.data() + n, g.data() + n + step_counter_ * n);
      y_res = torsten::precomputed_gradients(g, ode.ts_);
    }

    /**
     * When only @c ts is @c var, we don't solve
     * sensitivity ODE since the sensitivity is simply the RHS.
     */
    template<typename ode_type = Ode>
    inline std::enable_if_t<has_var_ts<ode_type>::value && (!has_var_y0<ode_type>::value) && (!has_var_par<ode_type>::value)>
    observer_impl(state_t& y_res, const N_Vector & y, const N_Vector* ys) const {
      auto g(ode_type::null_dbl_state(ode.ts_.size()));
      dbl_state_t dydt = ode.dbl_rhs_impl(curr_t_, y);
      for (size_t j = 0; j < n; ++j) {
        g[step_counter_] = dydt[j];
        // FIXME: use ts[i] instead of ts
        y_res[j] = stan::math::precomputed_gradients(NV_Ith_S(y, j), ode.ts_, g);
        g[step_counter_] = 0.0;
      }
    }

    /**
     * Only @c theta and/or init is @c var
     */
    template<typename ode_type = Ode>
    inline std::enable_if_t<(!has_var_ts<ode_type>::value) && (has_var_y0<ode_type>::value || has_var_par<ode_type>::value)>
    observer_impl(state_t& y_res, const dbl_state_t& y) const {
      y_res = torsten::precomputed_gradients(y, ode.vars());
    }

    /**
     * Only @c theta and/or init is @c var
     */
    template<typename ode_type = Ode>
    inline std::enable_if_t<(!has_var_ts<ode_type>::value) && (has_var_y0<ode_type>::value || has_var_par<ode_type>::value)>
    observer_impl(state_t& y_res, const N_Vector& y, const N_Vector* ys) const {
      y_res = torsten::precomputed_gradients(y, ys, ode.vars());
    }

    /**
     * @c theta and/or &c y0 are @c var, together with @c ts.
     */
    template<typename ode_type = Ode>
    inline std::enable_if_t<has_var_ts<ode_type>::value && (has_var_y0<ode_type>::value || has_var_par<ode_type>::value)>
    observer_impl(state_t& y_res, const dbl_state_t& y) const {
      auto g(ode_type::null_dbl_state(n * (1 + ns + ode.ts_.size())));
      std::copy(y.data(), y.data() + y.size(), g.data());        
      dbl_state_t dydt(n), y_dbl(n);
      std::copy(y.data(), y.data() + n, y_dbl.data());
      dydt = ode.dbl_rhs_impl(curr_t_, y_dbl);
      std::copy(dydt.data(), dydt.data() + n, g.data() + n + ns * n + step_counter_ * n);
      y_res = torsten::precomputed_gradients(g, ode.vars());
    }

    /**
     * @c theta and/or &c y0 are @c var, together with @c ts.
     */
    template<typename ode_type = Ode>
    inline std::enable_if_t<has_var_ts<ode_type>::value && (has_var_y0<ode_type>::value || has_var_par<ode_type>::value)>
    observer_impl(state_t& y_res,
                  const N_Vector& y,
                  const N_Vector* ys) const {
      auto g(ode_type::null_dbl_state(ns + ode.ts_.size()));
      dbl_state_t dydt = ode.dbl_rhs_impl(curr_t_, y);
      for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < ns; ++j) {
          g[j] = NV_Ith_S(ys[j], k);
        }
        g[step_counter_ + ns] = dydt[k];
        y_res[k] = precomputed_gradients(NV_Ith_S(y, k), ode.vars(), g);
        g[step_counter_ + ns] = 0.0;
      }
    }

    // /*
    //  * @c theta and/or &c y0 are @c var, together with @c ts.
    //  */
    // template<typename F, typename T_init, typename T_par,
    //          std::std::enable_if_t<torsten::is_var<T_init, T_par>::value>* = nullptr>
    // inline void observer_impl(state_t& y_res,
    //                           const dbl_state_t& y,
    //                           const PMXOdeSystem<F, stan::math::var, T_init, T_par>& system) const {
    //   std::vector<double> g(n * (1 + ns + system.ts_.size()), 0.0);
    //   std::copy(y.begin(), y.end(), g.begin());        
    //   std::vector<double> dydt(n), y_dbl(y.begin(), y.begin() + n);
    //   system.dbl_rhs_impl(curr_t_, y, dydt);
    //   std::copy(dydt.begin(), dydt.end(), g.begin() + n + ns * n + i * n);
    //   y_res = torsten::precomputed_gradients(g, ode_.vars());
    // }

  };

  template<typename Ode>
  struct OdeDataObserver {
    const Ode& ode;
    const size_t n;
    const size_t m;
    const size_t ns;
    const size_t nt;
    Eigen::MatrixXd y;
    int step_counter_;
    double curr_t_;

    OdeDataObserver(const Ode& ode0) :
      ode(ode0), n(ode.N), m(ode.M), ns(ode.ns), nt(ode.ts_.size()),
      y(Eigen::MatrixXd::Zero(ode.system_size + n*(Ode::is_var_ts ? nt : 0), nt)),
      step_counter_(0)
    {}

    inline void reset(const Ode& ode_) {
      step_counter_ = 0;
      y.setZero();
    }

    inline void operator()(const std::vector<double>& curr_result, double t) {
      curr_t_ = t;
      if(curr_t_ > ode.t0_) {
        observer_impl(y, curr_result, ode);
        step_counter_++;
      }
    }

    inline void operator()(const N_Vector& curr_y, const N_Vector* curr_ys, double t) {
      curr_t_ = t;
      if(curr_t_ > ode.t0_) {
        observer_impl(y, curr_y, curr_ys, ode);
        step_counter_++;
      }
    }

  private:
    /*
     * @c ts is data
     */
    template<typename F, typename T_init, typename T_par>
    inline void observer_impl(Eigen::MatrixXd& y_res,
                              const std::vector<double>& y,
                              const PMXOdeSystem<F, double, T_init, T_par>& system) const {
      y_res.col(step_counter_) = Eigen::VectorXd::Map(y.data(), system.system_size);
    }

    /**
     * @c ts is @c var
     */
    template<typename F, typename T_init, typename T_par>
    inline void observer_impl(Eigen::MatrixXd& y_res,
                              const std::vector<double>& y,
                              const PMXOdeSystem<F, stan::math::var, T_init, T_par>& system) const {
      for (size_t j = 0; j < system.system_size; ++j) y_res(j, step_counter_) = y[j];
      std::vector<double> y_tmp(y.begin(), y.begin() + n);
      std::vector<double> dydt = system.dbl_rhs_impl(curr_t_, y_tmp);
      for (size_t j = 0; j < n; ++j) {
        y_res(system.system_size + step_counter_ * n + j, step_counter_) = dydt[j];        
      }
    }

    /*
     * @c ts is @c var
     */
    template<typename F, typename Tt, typename T_init, typename T_par>
    inline void observer_impl(Eigen::MatrixXd& y_res,
                              const N_Vector& y,
                              const N_Vector* ys,
                              const PMXOdeSystem<F, Tt, T_init, T_par>& system) const {
      y_res.block(0, step_counter_, n, 1) = Eigen::VectorXd::Map(NV_DATA_S(y), n);

      if (system.use_fwd_sens) {
        for (size_t j = 0; j < ns; ++j) {
          y_res.block(n + j * n, step_counter_, n, 1) = Eigen::VectorXd::Map(NV_DATA_S(ys[j]), n);
        }
      }

      if (system.is_var_ts) {
        std::vector<double> dydt = system.dbl_rhs_impl(curr_t_, y);
        y_res.block(n + (step_counter_ + ns) * n, step_counter_, n, 1) =
          Eigen::VectorXd::Map(dydt.data(), n);
      }
    }
  };

}
}
#endif
