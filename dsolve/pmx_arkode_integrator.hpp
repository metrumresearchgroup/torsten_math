#ifndef STAN_MATH_TORSTEN_DSOLVE_ARKODE_INTEGRATOR_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ARKODE_INTEGRATOR_HPP

#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/torsten/mpi/precomputed_gradients.hpp>
#include <stan/math/torsten/dsolve/pmx_arkode_system.hpp>
#include <stan/math/torsten/dsolve/sundials_check.hpp>
#include <arkode/arkode.h>
#include <arkode/arkode_erkstep.h>
#include <arkode/arkode_butcher_erk.h>
#include <type_traits>

namespace torsten {
namespace dsolve {

/**
 * ARKode integrator.
 */
  struct PMXArkodeIntegrator {
    const double rtol_;
    const double atol_;
    const int64_t max_num_steps_;

    /*
     * Observer stores return data that is updated as requested.
     * Usually the returned is of @c array_2d @c var type.
     */
    template<typename Ode, bool GenVar>
    struct SolObserver {
      const Ode& ode_;
      const size_t n;
      const size_t m;
      std::vector<std::vector<typename Ode::scalar_type>> y;
      int step_counter_;

      SolObserver(const Ode& ode) :
        ode_(ode), n(ode.N_), m(ode.M_),
        y(ode.ts_.size(), std::vector<typename Ode::scalar_type>(ode.N_, 0.0)),
        step_counter_(0)
      {}

      /*
       * use observer to convert y value and gradient to var
       * results, if necessary.
       */
      inline void operator()(const N_Vector& nv_y, double t) {
        if(t > ode_.t0_) {
          observer_impl(nv_y, ode_.ts_, ode_.y0_, ode_.theta_);
          step_counter_++;
        }
      }

    private:
      /*
       * All data, return data
       */
      inline void observer_impl(const N_Vector& nv_y,
                                const std::vector<double>& ts,
                                const std::vector<double>& y0,
                                const std::vector<double>& theta) {
        std::copy(N_VGetArrayPointer(nv_y), N_VGetArrayPointer(nv_y) + n, y[step_counter_].begin());
      }

      /*
       * When only @c ts is @c var, we don't solve
       * sensitivity ODE since the sensitivity is simply the RHS.
       */
      inline void observer_impl(const N_Vector& nv_y,
                                const std::vector<stan::math::var>& ts,
                                const std::vector<double>& y0,
                                const std::vector<double>& theta) {
        int n = y0.size();
        std::vector<double> g(n * (1 + ts.size()), 0.0);
        std::copy(N_VGetArrayPointer(nv_y), N_VGetArrayPointer(nv_y) + n, g.begin());
        ode_.serv.user_data.eval_rhs(ts[step_counter_].val(), nv_y);
        std::copy(ode_.serv.user_data.fval.begin(), ode_.serv.user_data.fval.end(), g.begin() + n + step_counter_ * n);
        y[step_counter_] = torsten::precomputed_gradients(g, ts);
      }

      /*
       * Only @c theta is @c var
       */
      inline void observer_impl(const N_Vector& nv_y,
                                const std::vector<double>& ts,
                                const std::vector<double>& y0,
                                const std::vector<stan::math::var>& theta) {
        std::vector<double> yd(N_VGetArrayPointer(nv_y), N_VGetArrayPointer(nv_y) + ode_.fwd_system_size());
        y[step_counter_] = torsten::precomputed_gradients(yd, theta);
      }

      /*
       * only @c y0 is @c var
       */
      inline void observer_impl(const N_Vector& nv_y,
                                const std::vector<double>& ts,
                                const std::vector<stan::math::var>& y0,
                                const std::vector<double>& theta) {
        std::vector<double> yd(N_VGetArrayPointer(nv_y), N_VGetArrayPointer(nv_y) + ode_.fwd_system_size());
        y[step_counter_] = torsten::precomputed_gradients(yd, y0);
      }

      /*
       * @c y0 and @c theta are @c var
       */
      inline void observer_impl(const N_Vector& nv_y,
                                const std::vector<double>& ts,
                                const std::vector<stan::math::var>& y0,
                                const std::vector<stan::math::var>& theta) {
        std::vector<double> yd(N_VGetArrayPointer(nv_y), N_VGetArrayPointer(nv_y) + ode_.fwd_system_size());
        y[step_counter_] = torsten::precomputed_gradients(yd, ode_.vars());
      }

      /*
       * @c theta and/or &c y0 are @c var, together with @c ts.
       */
      template<typename T1, typename T2>
      inline void observer_impl(const N_Vector& nv_y,
                                const std::vector<stan::math::var>& ts,
                                const std::vector<T1>& y0,
                                const std::vector<T2>& theta) {
        int ns = ode_.ns;
        int n = y0.size();
        std::vector<double> g(n * (1 + ns + ts.size()), 0.0);
        std::copy(N_VGetArrayPointer(nv_y), N_VGetArrayPointer(nv_y) + ode_.fwd_system_size(), g.begin());
        std::vector<double> dy_dt(n);
        ode_.serv.user_data.eval_rhs(ts[step_counter_].val(), nv_y);
        std::copy(ode_.serv.user_data.fval.begin(), ode_.serv.user_data.fval.end(), g.begin() + n + ns * n + step_counter_ * n);
        y[step_counter_] = torsten::precomputed_gradients(g, ode_.vars());
      }
    };

    /*
     * Observer stores return data that is updated as requested.
     * For MPI results we need return data type so it can be
     * sent over to other nodes before reassembled into @c var
     * type. In this case, the returned matrix contain value
     * and sensitivity.
     */
    template<typename Ode>
    struct SolObserver<Ode, false> {
      Eigen::MatrixXd y;

      SolObserver(const Ode& ode) :
        y(Eigen::MatrixXd::Zero(ode.fwd_system_size() +
                                ode.n() * (Ode::is_var_ts ? ode.ts().size() : 0),
                                ode.ts().size()))
      {}
    };

  public:
    static constexpr int ARKODE_MAX_STEPS = 500;

    /**
     * constructor
     * @param[in] rtol relative tolerance
     * @param[in] atol absolute tolerance
     * @param[in] max_num_steps max nb. of times steps
     */
    PMXArkodeIntegrator(const double rtol, const double atol,
                        const int64_t max_num_steps = ARKODE_MAX_STEPS)
      : rtol_(rtol), atol_(atol), max_num_steps_(max_num_steps) {
      using stan::math::invalid_argument;
      if (rtol_ <= 0)
        invalid_argument("cvodes_integrator", "relative tolerance,", rtol_, "",
                         ", must be greater than 0");
      if (rtol_ > 1.0E-3)
        invalid_argument("cvodes_integrator", "relative tolerance,", rtol_, "",
                         ", must be less than 1.0E-3");
      if (atol_ <= 0)
        invalid_argument("cvodes_integrator", "absolute tolerance,", atol_, "",
                         ", must be greater than 0");
      if (max_num_steps_ <= 0)
        invalid_argument("cvodes_integrator", "max_num_steps,",
                         max_num_steps_, "",
                         ", must be greater than 0");
    }
      
    template<typename Ode>
    void solve(Ode& ode, SolObserver<Ode, true>& obs) {
      double t1 = ode.t0();

      auto mem       = ode.mem();
      auto y         = ode.nv_y();

      for (size_t i = 0; i < ode.ts().size(); ++i) {
        double t = stan::math::value_of(ode.ts()[i]);
        CHECK_SUNDIALS_CALL(ERKStepEvolve(mem, t, y, &t1, ARK_NORMAL));
        obs(ode.nv_y(), t1);
      }
    }

    /**
     * Return the solutions for the specified ODE
     * given the specified initial state,
     * initial times, times of desired solution, and parameters and
     * data, writing error and warning messages to the specified
     * stream contained in the ODE system.
     *
     * @tparam ODE type of ODE system
     * @param[in] ode ODE system
     * increasing order, all greater than the initial time.
     * @return a vector of states, each state being a vector of the
     * same size as the state variable, corresponding to a time in ts.
     */
    template <typename Ode, bool GenVar = true>
    auto integrate(Ode& ode) {
      using std::vector;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using Eigen::MatrixXd;

      auto mem       = ode.mem();
      auto y         = ode.nv_y();
      const size_t n = ode.n();
      const size_t ns= ode.ns();

      SolObserver<Ode, GenVar> observer(ode);

      /** Initial condition is from nv_y, which has changed
       * from previous solution, we need to reset it.
       * if y0 is parameter, the first n sensitivity vector
       * are regarding y0, thus they form a unit matrix.
       **/
      for (size_t i = 0; i < n; ++i) {
        NV_Ith_S(y, i) = ode.y0_d()[i];
        if (Ode::is_var_y0) {
          NV_Ith_S(y, n + n * i + i) = 1.0;
        }
      }

      ode.set_user_data(ode.serv);

      try {
        CHECK_SUNDIALS_CALL(ERKStepReInit(mem, ode.serv.arkode_rhs, ode.t0(), y));
        CHECK_SUNDIALS_CALL(ERKStepSStolerances(mem, rtol_, atol_));
        CHECK_SUNDIALS_CALL(ERKStepSetMaxNumSteps(mem, max_num_steps_));
        CHECK_SUNDIALS_CALL(ERKStepSetTableNum(mem, DORMAND_PRINCE_7_4_5));

        // the return type for MPI version is based on @c double,
        // as the return consists of the ARKode solution
        // instead of the assembled @c var vector in the
        // sequential version.
        solve(ode, observer);
      } catch (const std::exception& e) {
        throw;
      }

      return observer.y;
    }
  };  // cvodes integrator

}  // namespace dsolve
}  // namespace torsten

#endif
