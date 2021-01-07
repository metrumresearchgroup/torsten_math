#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_ARKODE_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_ARKODE_SYSTEM_HPP

#include <stan/math/rev/core/recover_memory.hpp>
#include <stan/math/torsten/dsolve/arkode_service.hpp>
#include <stan/math/torsten/dsolve/ode_forms.hpp>
#include <stan/math/prim/meta/return_type.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_vars.hpp>
#include <stan/math/prim/fun/get.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/err/check_size_match.hpp>
#include <stan/math/rev/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace torsten {
namespace dsolve {

  /**
   * Boost Odeint ODE system that contains informtion on residual
   * equation functor, sensitivity residual equation functor,
   * as well as initial conditions. This is a base type that
   * is intended to contain common values used by forward
   * sensitivity system.
   *
   * @tparam F type of functor for ODE residual
   * @tparam Tts scalar type of time steps
   * @tparam Ty0 scalar type of initial unknown values
   * @tparam Tpar scalar type of parameters
   */
  template <typename F, typename Tts, typename Ty0, typename Tpar>
  struct PMXArkodeSystem {
    using Ode = PMXArkodeSystem<F, Tts, Ty0, Tpar>;

    PMXOdeService<Ode>& serv;

    const F& f_;
    const double t0_;
    const std::vector<Tts>& ts_;
    const std::vector<Ty0>& y0_;
    const std::vector<Tpar>& theta_;
    const std::vector<double> y0_dbl_;
    const std::vector<double> theta_dbl_;
    const std::vector<double>& x_r_;
    const std::vector<int>& x_i_;
    const size_t N_;
    const size_t M_;
    const size_t ns_;  // nb. of sensi params
    const size_t size_;  // nb. of sensi params
    N_Vector& nv_y_;
    std::vector<double>& y_vec_;
    std::vector<double>& fval_;
    void* mem_;
    std::ostream* msgs_;

    static constexpr bool is_var_y0 = stan::is_var<Ty0>::value;
    static constexpr bool is_var_par = stan::is_var<Tpar>::value;
    static constexpr bool need_fwd_sens = is_var_y0 || is_var_par;

    // when ts is param, we don't have to do fwd
    // sensitivity by solving extra ODEs, because in this
    // case the sensitivity regarding ts is just the RHS
    // of ODE. We can append this type of sensitivity
    // results after ARKode solutions.
    static constexpr bool is_var_ts = stan::is_var<Tts>::value;

    using scalar_type = typename stan::return_type<Tts, Ty0, Tpar>::type;

    /** 
     * Constructor
     * 
     * @param[in] serv arkode service for memory management
     * @param[in] f rhs functor
     * @param[in] t0 initial time
     * @param[in] ts output time points
     * @param[in] y0 initial condition
     * @param[in] theta parameters
     * @param[in] x_r real data
     * @param[in] x_i integer data
     * @param[in] msgs output msg stream
     * 
     */
    template<typename ode_t>
    PMXArkodeSystem(dsolve::PMXOdeService<ode_t>& serv0,
                    const F& f,
                    double t0,
                    const std::vector<Tts>& ts,
                    const std::vector<Ty0>& y0,
                    const std::vector<Tpar>& theta,
                    const std::vector<double>& x_r,
                    const std::vector<int>& x_i,
                    std::ostream* msgs)
      : serv(serv0),
        f_(f),
        t0_(t0),
        ts_(ts),
        y0_(y0),
        theta_(theta),
        y0_dbl_(stan::math::value_of(y0)),
        theta_dbl_(stan::math::value_of(theta)),
        x_r_(x_r),
        x_i_(x_i),
        N_(y0.size()),
        M_(theta.size()),
        ns_((is_var_y0 ? N_ : 0) + (is_var_par ? M_ : 0)),
        size_(serv.user_data.fwd_ode_dim),
        nv_y_(serv.nv_y),
        y_vec_(serv.user_data.y),
        fval_(serv.user_data.fval),
        mem_(serv.mem),
        msgs_(msgs) {
      using stan::math::system_error;

      if (nv_y_ == NULL)
        throw std::runtime_error("N_VMake_Serial failed to allocate memory");

      if (mem_ == NULL)
        throw std::runtime_error("ERKStepCreate failed to allocate memory");

      static const char* caller = "PMXArkodeSystem";
      int err = 1;
      if (N_ != serv.user_data.N)
        system_error(caller, "N_", err, "inconsistent allocated memory");
      if (size_ != size_t(N_VGetLength_Serial(nv_y_)))
        system_error(caller, "nv_y", err, "inconsistent allocated memory");
      if (M_ != serv.user_data.M)
        system_error(caller, "M_", err, "inconsistent allocated memory");
      if (ns_ != serv.user_data.ns)
        system_error(caller, "ns_", err, "inconsistent allocated memory");

      // initial condition
      for (size_t i = 0; i < N_; ++i) {
        NV_Ith_S(nv_y_, i) = y0_dbl_[i];
      }
    }

    /**
     * destructor is empty as all ARKode resources are
     * handled by @c PMXOdeService
     */
    ~PMXArkodeSystem() {}

    /**
     * Return the size of the solution correspdoning each y[i],
     * namely 1 + number of sensitivities.
     *
     * @return each solution size.
     */
    int n_sol() const {
      int nt = stan::is_var<Tts>::value ? ts_.size() : 0;
      return 1 + ns_ + nt;
    }

    /**
     * return reference to initial time
     *
     * @return reference to initial time
     */
    const double& t0() { return t0_; }

    /**
     * return reference to time steps
     *
     * @return reference to time steps
     */
    const std::vector<Tts> & ts() const { return ts_; }

    /**
     * return reference to current N_Vector of unknown variable
     *
     * @return reference to current N_Vector of unknown variable
     */
    N_Vector& nv_y() { return nv_y_; }

    /**
     * return reference to initial condition
     *
     * @return reference to initial condition
     */
    inline const std::vector<Ty0>& y0() const { return y0_; }

    /**
     * return reference to initial condition data
     *
     * @return reference to initial condition data
     */
    inline const std::vector<double>& y0_d() const { return y0_dbl_; }

    /**
     * return reference to parameter
     *
     * @return reference to parameter
     */
    inline const std::vector<Tpar>& theta() const { return theta_; }

    /*
     * retrieving a vector of vars that will be used as parameters
     */
    inline auto vars() const {
      return pmx_ode_vars(y0_, theta_, ts_);
    }

    /**
     * return current RHS evaluation.
     */
    const std::vector<double>& fval() { return fval_; }

    /**
     * return number of unknown variables
     */
    const size_t n() const { return N_; }

    /**
     * return number of sensitivity parameters
     */
    const size_t ns() { return ns_; }

    /**
     * return size of ODE system for primary and sensitivity unknowns
     */
    inline const size_t fwd_system_size() const { return size_; }

    /**
     * return theta size
     */
    const size_t n_par() { return theta_.size(); }

    /**
     * return CVODES memory handle
     */
    void* mem() { return mem_; }

    /**
     * return reference to ODE functor
     */
    const F& f() const { return f_; }

    /** 
     * Set user_data items using runtime ODE info, so that when <code>user_data</code>
     * is passed in runtime functions it provides correct context.
     *
     * @param serv <code>cvodes_serv</code> object that stores <code>user_data</code>
     */
    template<typename ode_t>
    void set_user_data(PMXOdeService<ode_t>& serv) {
      serv.user_data.f = &this -> f_;
      serv.user_data.theta_d = stan::math::value_of(theta_);
      serv.user_data.px_r = &this -> x_r_;
      serv.user_data.px_i = &this -> x_i_;
      serv.user_data.msgs = msgs_;
    }
  };

}  // namespace dsolve
}  // namespace torsten
#endif
