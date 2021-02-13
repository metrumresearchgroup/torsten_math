#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_ODE_INTEGRATOR_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_ODE_INTEGRATOR_HPP

#include <stan/math/prim/fun/eval.hpp>
#include <stan/math/rev/functor/jacobian.hpp>
#include <stan/math/prim/meta/return_type.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/integrate_ode_rk45.hpp>
#include <stan/math/rev/functor/integrate_ode_adams.hpp>
#include <stan/math/rev/functor/integrate_ode_bdf.hpp>
#include <stan/math/torsten/dsolve/cvodes_service.hpp>
#include <stan/math/torsten/dsolve/pmx_cvodes_system.hpp>
#include <stan/math/torsten/dsolve/pmx_cvodes_integrator.hpp>
#include <stan/math/torsten/dsolve/pmx_odeint_integrator.hpp>
#include <stan/math/torsten/dsolve/ode_check.hpp>
#include <ostream>
#include <vector>

namespace torsten {
  namespace dsolve {
    template<template<typename...> class ode_type, typename integrator_t>
    struct PMXOdeIntegrator {
      const double rtol;
      const double atol;
      const long int max_num_step;
      const double as_rtol;
      const double as_atol;
      const long int as_max_num_step;
      std::ostream* msgs;

      PMXOdeIntegrator(double rtol0,
                       double atol0,
                       long int max_num_step0,
                       double as_rtol0,
                       double as_atol0,
                       long int as_max_num_step0,
                       std::ostream* msgs0) :
        rtol(rtol0), atol(atol0), max_num_step(max_num_step0),
        as_rtol(as_rtol0), as_atol(as_atol0), as_max_num_step(as_max_num_step0),
        msgs(msgs0)
      {}
      
      PMXOdeIntegrator(double rtol0, double atol0, long int max_num_step0,
                       std::ostream* msgs0) :
        rtol(rtol0), atol(atol0), max_num_step(max_num_step0),
        as_rtol(1e-6), as_atol(1e-6), as_max_num_step(100),
        msgs(msgs0)
      {}

      PMXOdeIntegrator() :
        rtol(1e-10), atol(1e-10), max_num_step(1e8),
        as_rtol(1e-6), as_atol(1e-6), as_max_num_step(100),
        msgs(0)
      {}

      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type_t<Tt, T_initial, T_param>> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i) const {
        static const char* caller = "pmx_integrate_ode";
        dsolve::ode_check(y0, t0, ts, theta, x_r, x_i, caller);

        const int m = theta.size();
        const int n = y0.size();
        using Ode = ode_type<F, Tt, T_initial, T_param>;
        dsolve::PMXOdeService<Ode> serv(n, m);
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
        integrator_t solver(rtol, atol, max_num_step);
        return solver.integrate(ode);
      }

      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type_t<Tt, T_initial, T_param>> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const Tt& t1,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i) const {
        std::vector<Tt> ts{t1};
        return (*this)(f, y0, t0, ts, theta, x_r, x_i);
      }

      template <typename F, typename Tt, typename T_initial, typename T_param>
      Eigen::MatrixXd
      solve_d(const F& f,
              const std::vector<T_initial>& y0,
              double t0,
              const std::vector<Tt>& t1,
              const std::vector<T_param>& theta,
              const std::vector<double>& x_r,
              const std::vector<int>& x_i) const {
        std::vector<Tt> ts{t1};
        static const char* caller = "pmx_integrate_ode";
        dsolve::ode_check(y0, t0, ts, theta, x_r, x_i, caller);

        const int m = theta.size();
        const int n = y0.size();
        using Ode = ode_type<F, Tt, T_initial, T_param>;
        dsolve::PMXOdeService<Ode> serv(n, m);
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
        integrator_t solver(rtol, atol, max_num_step);
        return solver. template integrate<Ode, false>(ode);
      }
    };

    template<typename scheme_t>
    struct PMXOdeIntegrator<dsolve::PMXOdeintSystem, PMXOdeintIntegrator<scheme_t>> {
      const double rtol;
      const double atol;
      const long int max_num_step;
      const double as_rtol;
      const double as_atol;
      const long int as_max_num_step;
      std::ostream* msgs;

      PMXOdeIntegrator(double rtol0,
                       double atol0,
                       long int max_num_step0,
                       double as_rtol0,
                       double as_atol0,
                       long int as_max_num_step0,
                       std::ostream* msgs0) :
        rtol(rtol0), atol(atol0), max_num_step(max_num_step0),
        as_rtol(as_rtol0), as_atol(as_atol0), as_max_num_step(as_max_num_step0),
        msgs(msgs0)
      {}
      
      PMXOdeIntegrator(double rtol0, double atol0, long int max_num_step0,
                       std::ostream* msgs0) :
        rtol(rtol0), atol(atol0), max_num_step(max_num_step0),
        as_rtol(1e-6), as_atol(1e-6), as_max_num_step(100),
        msgs(msgs0)
      {}

      PMXOdeIntegrator() :
        rtol(1e-10), atol(1e-10), max_num_step(1e8),
        as_rtol(1e-6), as_atol(1e-6), as_max_num_step(100),
        msgs(0)
      {}

      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type_t<Tt, T_initial, T_param>> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i) const {
        static const char* caller = "pmx_integrate_ode";
        dsolve::ode_check(y0, t0, ts, theta, x_r, x_i, caller);

        using Ode = dsolve::PMXOdeintSystem<F, Tt, T_initial, T_param>;
        const int m = theta.size();
        const int n = y0.size();

        dsolve::PMXOdeService<Ode> serv(n, m);
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
        dsolve::PMXOdeintIntegrator<scheme_t> solver(rtol, atol, max_num_step);
        return solver.integrate(ode);
      }

      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type_t<Tt, T_initial, T_param>> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const Tt& t1,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i) const {
        std::vector<Tt> ts{t1};
        return (*this)(f, y0, t0, ts, theta, x_r, x_i);
      }

      template <typename F, typename Tt, typename T_initial, typename T_param>
      Eigen::MatrixXd
      solve_d(const F& f,
              const std::vector<T_initial>& y0,
              double t0,
              const std::vector<Tt>& ts,
              const std::vector<T_param>& theta,
              const std::vector<double>& x_r,
              const std::vector<int>& x_i) const {
        using Ode = torsten::dsolve::PMXOdeintSystem<F, Tt, T_initial, T_param>;
        const int m = theta.size();
        const int n = y0.size();

        dsolve::PMXOdeService<Ode> serv(n, m);
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
        dsolve::PMXOdeintIntegrator<scheme_t> solver(rtol, atol, max_num_step);
        return solver.template integrate<Ode, false>(ode);
      }
    };

    /**
     * Dummy integrator type for analytical solutions such as one/two cpt.
     * 
     */
    struct PMXAnalyiticalIntegrator {};
  }

  template<typename integrator_type>
  struct has_data_only_output {
    static const bool value = false;
  };

  template<typename integrator_t>
  struct has_data_only_output<dsolve::PMXOdeIntegrator<dsolve::PMXOdeintSystem, integrator_t>> {
    static const bool value = true;
  };

  template<typename integrator_t>
  struct has_data_only_output<dsolve::PMXOdeIntegrator<dsolve::PMXCvodesFwdSystem_adams, integrator_t>> {
    static const bool value = true;
  };

  template<typename integrator_t>
  struct has_data_only_output<dsolve::PMXOdeIntegrator<dsolve::PMXCvodesFwdSystem_bdf, integrator_t>> {
    static const bool value = true;
  };
}

#endif
