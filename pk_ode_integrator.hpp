#ifndef TORSTEN_ODE_INTEGRATOR_HPP
#define TORSTEN_ODE_INTEGRATOR_HPP

#include <stan/math/torsten/dsolve/dsolve.hpp>
#include <stan/math/rev/mat.hpp>
#include <ostream>
#include <vector>

namespace torsten {
  enum PkOdeIntegratorId {
    Analytical,
    StanRk45, StanAdams, StanBdf,
    PkAdams, PkBdf
  };
}

namespace torsten {
  namespace internal {
    template<PkOdeIntegratorId It>
    struct PkOdeIntegratorDispatcher;

    /* 
     * specification for a Stan's ODE integrator. Since @c
     * ts cannot be param in Stan's integrators, we convert
     * it to data first.
     */
    template<>
    struct PkOdeIntegratorDispatcher<StanRk45> {

      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type<T_initial, T_param>::type> > // NOLINT
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i,
                 std::ostream* msgs,
                 double rtol,
                 double atol,
                 long int max_num_step) {  // NOLINT(runtime/int)                   
        const std::vector<double>& ts_d(stan::math::value_of(ts));
        return stan::math::integrate_ode_rk45(f, y0, t0, ts_d, theta,
                                              x_r, x_i, msgs, rtol,
                                              atol, max_num_step);
      }
    };

    /* 
     * specification for a Stan's ODE integrator. Since @c
     * ts cannot be param in Stan's integrators, we convert
     * it to data first.
     */
    template<>
    struct PkOdeIntegratorDispatcher<StanAdams> {
      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type<T_initial,
                                                         T_param>::type> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i,
                 std::ostream* msgs,
                 double rtol,
                 double atol,
                 long int max_num_step) {  // NOLINT(runtime/int)                   
        const std::vector<double>& ts_d(stan::math::value_of(ts));
        return stan::math::integrate_ode_adams(f, y0, t0, ts_d, theta,
                                               x_r, x_i, msgs, rtol,
                                               atol, max_num_step);
      }
    };

    /* 
     * specification for a Stan's ODE integrator. Since @c
     * ts cannot be param in Stan's integrators, we convert
     * it to data first.
     */
    template<>
    struct PkOdeIntegratorDispatcher<StanBdf> {
      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type<T_initial,
                                                         T_param>::type> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i,
                 std::ostream* msgs,
                 double rtol,
                 double atol,
                 long int max_num_step) {  // NOLINT(runtime/int)                   
        const std::vector<double>& ts_d(stan::math::value_of(ts));
        return stan::math::integrate_ode_bdf(f, y0, t0, ts_d, theta,
                                             x_r, x_i, msgs, rtol,
                                             atol, max_num_step);
      }
    };

    /* 
     * specification for a Torsten's ODE integrator, in
     * which @c ts can be a param.
     */
    template<>
    struct PkOdeIntegratorDispatcher<PkAdams> {
      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type<Tt, T_initial,
                                                         T_param>::type> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i,
                 std::ostream* msgs,
                 double rtol,
                 double atol,
                 long int max_num_step) {  // NOLINT(runtime/int)                   
        return torsten::dsolve::pk_integrate_ode_adams(f, y0, t0, ts, theta,
                                                       x_r, x_i, msgs, rtol,
                                                       atol, max_num_step);
      }
    };

    /* 
     * specification for a Torsten's ODE integrator, in
     * which @c ts can be a param.
     */
    template<>
    struct PkOdeIntegratorDispatcher<PkBdf> {
      template <typename F, typename Tt, typename T_initial, typename T_param>
      std::vector<std::vector<typename stan::return_type<Tt, T_initial,
                                                         T_param>::type> >
      operator()(const F& f,
                 const std::vector<T_initial>& y0,
                 double t0,
                 const std::vector<Tt>& ts,
                 const std::vector<T_param>& theta,
                 const std::vector<double>& x_r,
                 const std::vector<int>& x_i,
                 std::ostream* msgs,
                 double rtol,
                 double atol,
                 long int max_num_step) {  // NOLINT(runtime/int)                   
        return torsten::dsolve::pk_integrate_ode_bdf(f, y0, t0, ts, theta,
                                                     x_r, x_i, msgs, rtol,
                                                     atol, max_num_step);
      }
    };    
  }
}

namespace torsten {
  template<PkOdeIntegratorId It>
  struct PkOdeIntegrator {
    const double rtol;
    const double atol;
    const long int max_num_step;
    std::ostream* msgs;

    PkOdeIntegrator() : rtol(1e-10), atol(1e-10), max_num_step(1e8), msgs(0) {}

    PkOdeIntegrator(const double rtol0, const double atol0,
                    const long int max_num_step0,
                    std::ostream* msgs0) :
      rtol(rtol0), atol(atol0), max_num_step(max_num_step0), msgs(msgs0)
    {}
    
    /*
     * Stan's ODE solver function doesn't support @c T_t to
     * be @c var
     */
    template <typename F, typename Tt, typename T_initial, typename T_param>
    std::vector<std::vector<typename stan::return_type<T_initial, T_param>::type> > // NOLINT
    operator()(const F& f,
               const std::vector<T_initial>& y0,
               double t0,
               const std::vector<Tt>& ts,
               const std::vector<T_param>& theta,
               const std::vector<double>& x_r,
               const std::vector<int>& x_i) const {
      using internal::PkOdeIntegratorDispatcher;
      return PkOdeIntegratorDispatcher<It>()(f, y0, t0, ts, theta, x_r, x_i,
                                             msgs, rtol, atol, max_num_step);
    }
  };

  /*
   * specialization for @c PkBdf
   */
  template<>
  struct PkOdeIntegrator<PkBdf> {
    const double rtol;
    const double atol;
    const long int max_num_step;
    std::ostream* msgs;

    PkOdeIntegrator() : rtol(1e-10), atol(1e-10), max_num_step(1e8), msgs(0) {}

    PkOdeIntegrator(const double rtol0, const double atol0,
                    const long int max_num_step0,
                    std::ostream* msgs0) :
      rtol(rtol0), atol(atol0), max_num_step(max_num_step0), msgs(msgs0)
    {}

    /*
     * Torsten's ODE solvers support @c T_t to be @c var
     */
    template <typename F, typename Tt, typename T_initial, typename T_param>
    std::vector<std::vector<typename stan::return_type<Tt, T_initial, T_param>::type> > // NOLINT
    operator()(const F& f,
               const std::vector<T_initial>& y0,
               double t0,
               const std::vector<Tt>& ts,
               const std::vector<T_param>& theta,
               const std::vector<double>& x_r,
               const std::vector<int>& x_i) const {
      using internal::PkOdeIntegratorDispatcher;
      return PkOdeIntegratorDispatcher<PkBdf>()(f, y0, t0, ts, theta, x_r, x_i,
                                                msgs, rtol, atol, max_num_step);
    }

    /*
     * For MPI solution we need to return data that consists
     * of solution value and gradients
     */
    template <typename F, typename Tt, typename T_initial, typename T_param>
    Eigen::MatrixXd
    solve_d(const F& f,
            const std::vector<T_initial>& y0,
            double t0,
            const std::vector<Tt>& ts,
            const std::vector<T_param>& theta,
            const std::vector<double>& x_r,
            const std::vector<int>& x_i) const {
      using Ode = torsten::dsolve::PKCvodesFwdSystem<F, Tt, T_initial, T_param, CV_BDF, AD>;
      const int m = theta.size();
      const int n = y0.size();

      dsolve::PKCvodesService<typename Ode::Ode> serv(n, m);

      Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};

      torsten::dsolve::PKCvodesIntegrator solver(rtol, atol, max_num_step);
      Eigen::MatrixXd res = solver.integrate<Ode, false>(ode);

      return res;
    }
  };

  /*
   * specialization for @c PkAdams
   */
  template<>
  struct PkOdeIntegrator<PkAdams> {
    const double rtol;
    const double atol;
    const long int max_num_step;
    std::ostream* msgs;

    PkOdeIntegrator() : rtol(1e-10), atol(1e-10), max_num_step(1e8), msgs(0) {}

    PkOdeIntegrator(const double rtol0, const double atol0,
                    const long int max_num_step0,
                    std::ostream* msgs0) :
      rtol(rtol0), atol(atol0), max_num_step(max_num_step0), msgs(msgs0)
    {}

    /*
     * Torsten's ODE solvers support @c T_t to be @c var
     */
    template <typename F, typename Tt, typename T_initial, typename T_param>
    std::vector<std::vector<typename stan::return_type<Tt, T_initial, T_param>::type> > // NOLINT
    operator()(const F& f,
               const std::vector<T_initial>& y0,
               double t0,
               const std::vector<Tt>& ts,
               const std::vector<T_param>& theta,
               const std::vector<double>& x_r,
               const std::vector<int>& x_i) const {
      using internal::PkOdeIntegratorDispatcher;
      return PkOdeIntegratorDispatcher<PkAdams>()(f, y0, t0, ts, theta, x_r, x_i,
                                                msgs, rtol, atol, max_num_step);
    }

    /*
     * For MPI solution we need to return data that consists
     * of solution value and gradients
     */
    template <typename F, typename Tt, typename T_initial, typename T_param>
    Eigen::MatrixXd
    solve_d(const F& f,
            const std::vector<T_initial>& y0,
            double t0,
            const std::vector<Tt>& ts,
            const std::vector<T_param>& theta,
            const std::vector<double>& x_r,
            const std::vector<int>& x_i) const {
      using Ode = torsten::dsolve::PKCvodesFwdSystem<F, Tt, T_initial, T_param, CV_ADAMS, AD>;
      const int m = theta.size();
      const int n = y0.size();

      dsolve::PKCvodesService<typename Ode::Ode> serv(n, m);

      Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};

      torsten::dsolve::PKCvodesIntegrator solver(rtol, atol, max_num_step);
      Eigen::MatrixXd res = solver.integrate<Ode, false>(ode);

      return res;
    }
  };
}

#endif
