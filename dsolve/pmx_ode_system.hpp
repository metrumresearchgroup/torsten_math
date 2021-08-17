#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_ODE_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_ODE_SYSTEM_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta/return_type.hpp>
#include <stan/math/prim/functor/for_each.hpp>
#include <stan/math/torsten/dsolve/ode_tuple_functor.hpp>
#include <stan/math/torsten/dsolve/ode_check.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_vars.hpp>
#include <stan/math/torsten/meta/require_generics.hpp>
#include <stan/math/torsten/value_of.hpp>
#include <stan/math/torsten/dsolve/braid_data.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/rev/core/typedefs.hpp>
#include <stan/math/rev/fun/to_var.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/rev/fun/value_of_rec.hpp>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <arkode/arkode.h>
#include <arkode/arkode_erkstep.h>
#include <ostream>
#include <stdexcept>
#include <vector>

#ifdef TORSTEN_BRAID
#include <arkode/arkode_xbraid.h>
#include <braid.h>
#include <boost/mpi.hpp>
#endif

namespace torsten {
namespace dsolve {
  struct deep_copy_tuple {
    template<typename Tuple>
    auto operator()(Tuple const& t) {
      static constexpr auto size = std::tuple_size<Tuple>::value;
      return copy_impl(t, std::make_index_sequence<size>{});
    }

  private:
    template<typename Tuple, size_t ... I>
    auto copy_impl(Tuple const& t, std::index_sequence<I ...>) {
      using stan::math::deep_copy_vars;
      std::tuple<decltype(deep_copy_vars(std::get<I>(t)))...>
        new_tuple(deep_copy_vars(std::get<I>(t))...);
      return new_tuple;
    }
  };

  /**
   * ODE system that contains informtion on residual
   * equation functor, sensitivity residual equation functor,
   * as well as initial conditions. This is a base type that
   * is intended to contain common values used by forward
   * sensitivity system.
   *
   * @tparam F type of functor for ODE residual
   * @tparam Tt scalar type of time steps
   * @tparam T_init scalar type of initial unknown values, specified
   *         as std::vector<T_init>
   * @tparam T_par scalar type of parameters
   */
  template <typename F, typename Tt, typename T_init, typename T_par>
  struct PMXOdeSystem : torsten::std_ode<F> {
    using Ode = PMXOdeSystem<F, Tt, T_init, T_par>;
    using scalar_t = typename stan::return_type_t<Tt, T_init, T_par>;
    using state_t = std::vector<scalar_t>;
    static constexpr bool is_var_ts  = stan::is_var<Tt>::value;
    static constexpr bool is_var_y0  = stan::is_var<T_init>::value;
    static constexpr bool is_var_par = stan::is_var<T_par>::value;
    static constexpr bool use_fwd_sens = is_var_y0 || is_var_par;

    const F& f_;
    const double t0_;
    const std::vector<Tt>& ts_;
    const std::vector<T_init>& y0_;
    const std::vector<T_par>& theta_;
    const std::vector<double> theta_dbl_;
    const std::vector<double>& x_r_;
    const std::vector<int>& x_i_;
    const size_t N;
    const size_t M;
    const size_t ns;
    const size_t system_size;
    std::ostream* msgs_;
    std::vector<double> y0_fwd_system;
    std::vector<double> y_dbl_work, dydt_dbl_work;
  public:
    PMXOdeSystem(const F& f,
                 double t0,
                 const std::vector<Tt>& ts,
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
        theta_dbl_(stan::math::value_of(theta)),
        x_r_(x_r),
        x_i_(x_i),
        N(y0.size()),
        M(theta.size()),      
        ns((is_var_y0 ? N : 0) + (is_var_par ? M : 0)),
        system_size(N + N * ns),
        msgs_(msgs),
        y0_fwd_system(system_size, 0.0),
        y_dbl_work(N),
        dydt_dbl_work(N)
    {
      const char* caller = "PMX ODE System";
      torsten::dsolve::ode_check(y0_, t0_, ts_, theta_, x_r_, x_i_, caller);

      // initial state
      std::transform(y0.begin(), y0.end(), y0_fwd_system.begin(),
                     [](const T_init& v){ return stan::math::value_of(v); });
      if (is_var_y0)  {
        for (size_t i = 0; i < N; ++i) {
          y0_fwd_system[N + i * N + i] = 1.0;        
        }
      }
    }

    static std::vector<double> null_dbl_state(size_t n_size) {
      return std::vector<double>(n_size, 0.0);
    }

    static state_t null_state(size_t n_size) {
      return std::vector<scalar_t>(n_size, 0.0);
    }

    /*
     * retrieving a vector of vars that will be used as parameters
     */
    inline auto vars() const {
      return pmx_ode_vars(y0_, theta_, ts_);
    }

    /*
     * Evaluate RHS of the ODE(the combined system)
     * @param y current dependent value, arranged as {y, dy_dp1, dy_dp2...}
     * @param dy_dt ODE RHS to be filled.
     * @param t current indepedent value
     */
    inline void operator()(const std::vector<double>& y, std::vector<double>& dy_dt,
                           double t) {
      stan::math::check_size_match("PMXOdeSystem", "y", y.size(), "dy_dt", dy_dt.size());
      rhs_impl(y, dy_dt, t);
    }

    /**
     * evaluate RHS with data only inputs.
     */
    inline std::vector<double>& dbl_rhs_impl(double t, const std::vector<double>& y)
    {
      dydt_dbl_work = f_(t, y, theta_dbl_, x_r_, x_i_, msgs_);
      return dydt_dbl_work;
    }

    /**
     * evaluate RHS with data only inputs.
     */
    inline std::vector<double>& dbl_rhs_impl(double t, const N_Vector& nv_y)
    {
      for (int i = 0; i < N; ++i) {
        y_dbl_work[i] = NV_Ith_S(nv_y, i);
      }
      return dbl_rhs_impl(t, y_dbl_work);
    }

    /**
     * evaluate RHS with data only inputs for N_Vector data
     */    
    inline void operator()(double t, N_Vector& nv_y, N_Vector& ydot) {
      stan::math::check_size_match("PMXOdeSystem", "y", NV_LENGTH_S(nv_y), "dy_dt", NV_LENGTH_S(ydot));
      for (int i = 0; i < N; ++i) {
        y_dbl_work[i] = NV_Ith_S(nv_y, i);
      }
      std::vector<double>& dydt = dbl_rhs_impl(t, y_dbl_work);
      for (size_t i = 0; i < N; ++i) {
        NV_Ith_S(ydot, i) = dydt[i];
      }
    }

    static int cvodes_rhs(double t, N_Vector y, N_Vector ydot, void* user_data) {
      Ode* ode = static_cast<Ode*>(user_data);
      (*ode)(t, y, ydot);
      return 0;
    }

    /*
     * evalute RHS of the entire system, possibly including
     * the forward sensitivity equation components in @c y and @c dy_dt.
     */
    void rhs_impl(const std::vector<double>& y,
                  std::vector<double>& dy_dt, double t) {
      using std::vector;
      using stan::math::var;

      if (!(is_var_y0 || is_var_par)) {
        dy_dt = dbl_rhs_impl(t, y);
        return;
      }

      std::fill(dy_dt.begin(), dy_dt.end(), 0.0);
      stan::math::nested_rev_autodiff nested;

      std::vector<var> yv(y.begin(), y.begin() + N);
      std::vector<var> theta_v(theta_dbl_.begin(), theta_dbl_.end());
      std::vector<var> fyv(is_var_par ?
                           f_(t, yv, theta_v, x_r_, x_i_, msgs_) :
                           f_(t, yv, theta_, x_r_, x_i_, msgs_));

      stan::math::check_size_match("PMXOdeSystem", "dz_dt", fyv.size(), "states", N);

      for (size_t i = 0; i < N; ++i) {
        if (i > 0) {
          nested.set_zero_all_adjoints();            
        }
        dy_dt[i] = fyv[i].val();
        fyv[i].grad();

        // df/dy*s_i term, for i = 1...ns
        for (size_t j = 0; j < ns; ++j) {
          for (size_t k = 0; k < N; ++k) {
            dy_dt.at(N + N * j + i) += y[N + N * j + k] * yv[k].adj();
          }
        }

        // df/dp_i term, for i = n...n+m-1
        if (is_var_par) {
          for (size_t j = 0; j < M; ++j) {
            dy_dt.at(N + N * (ns - M + j) + i) += theta_v[j].adj();
          }
        }
      }
    }

    /**
     * Calculate sensitivity rhs using CVODES vectors. The
     * internal workspace is allocated by @c CvodesService.
     */
    inline void operator()(int ns, double t, N_Vector nv_y, N_Vector ydot,
                           N_Vector* ys, N_Vector* ysdot,
                           N_Vector temp1, N_Vector temp2) {
      // initialize ysdot
      for (int i = 0; i < ns; ++i) N_VConst(0.0, ysdot[i]);

      stan::math::nested_rev_autodiff nested;

      std::vector<stan::math::var> yv(NV_DATA_S(nv_y), NV_DATA_S(nv_y) + N);
      std::vector<stan::math::var> theta_v(theta_dbl_.begin(), theta_dbl_.end());
      std::vector<stan::math::var> fyv(is_var_par ?
                                         f_(t, yv, theta_v, x_r_, x_i_, msgs_) :
                                         f_(t, yv, theta_dbl_, x_r_, x_i_, msgs_));

      stan::math::check_size_match("PMXOdeSystem", "dy_dt", fyv.size(), "states", N);

      for (int j = 0; j < N; ++j) {
        if (j > 0) {
          nested.set_zero_all_adjoints();
        }
        fyv[j].grad();

        // df/dy*s_i term, for i = 1...ns
        for (int i = 0; i < ns; ++i) {
          auto ysp = N_VGetArrayPointer(ys[i]);
          auto nvp = N_VGetArrayPointer(ysdot[i]);
          for (int k = 0; k < N; ++k) {
            nvp[j] += yv[k].adj() * ysp[k];
          }
        }

        // df/dp_i term, for i = n...n+m-1
        if (is_var_par) {
          for (int i = 0; i < M; ++i) {
            auto nvp = N_VGetArrayPointer(ysdot[ns - M + i]);
            nvp[j] += theta_v[i].adj();
          }
        }
      }
    }

    static int cvodes_sens_rhs(int ns, double t, N_Vector y, N_Vector ydot,
                               N_Vector* ys, N_Vector* ysdot, void* user_data,
                               N_Vector temp1, N_Vector temp2) {
      if (use_fwd_sens) {
        Ode* ode = static_cast<Ode*>(user_data);
        (*ode)(ns, t, y, ydot, ys, ysdot, temp1, temp2);            
      }
      return 0;
    }

    /**
     * return a closure for CVODES residual callback using a
     * non-capture lambda.
     *
     * @tparam Ode type of Ode
     * @return RHS function for Cvodes
     */
    inline CVLsJacFn cvodes_jac() {
      return [](realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void* user_data, // NOLINT
                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) -> int {
        Ode* ode = static_cast<Ode*>(user_data);
        ode -> jac(t, y, fy, J);
        return 0;
      };
    }

    /**
     * evaluate Jacobian matrix using current state, store
     * the result in @c SUNMatrix J.
     *
     * @param t current time
     * @param y current y
     * @param fy current f(y)
     * @param J Jacobian matrix J(i,j) = df_i/dy_j
     */
    inline void jac(double t, N_Vector& nv_y, N_Vector& fy, SUNMatrix& J) {
      stan::math::nested_rev_autodiff nested;

      std::vector<stan::math::var> yv_work(NV_DATA_S(nv_y), NV_DATA_S(nv_y) + N);
      std::vector<stan::math::var> fyv_work(f_(t, yv_work, theta_dbl_, x_r_, x_i_, msgs_));

      for (int i = 0; i < N; ++i) {
        nested.set_zero_all_adjoints();
        fyv_work[i].grad();
        for (int j = 0; j < N; ++j) {
          SM_ELEMENT_D(J, i, j) = yv_work[j].adj();
        }
      }
    }
  };

  /**
   * ODE system that contains informtion on residual
   * equation functor, sensitivity residual equation functor,
   * as well as initial conditions. This is a base type that
   * is intended to contain common values used by forward
   * sensitivity system.
   *
   * @tparam F type of functor for ODE residual
   * @tparam Tt scalar type of time steps
   * @tparam T_init scalar type of initial unknown values
   * @tparam T_par variadic parameters
   */
  template <typename F, typename Tt, typename T_init, typename... T_par>
  class PMXVariadicOdeSystem : torsten::eigen_ode<F, T_par...> {
    using Ode = PMXVariadicOdeSystem<F, Tt, T_init, T_par...>;

    const F& f_;
    const TupleOdeFunc<F> f_tuple_;
    std::tuple<decltype(torsten::value_of(std::declval<const T_par&>()))...> theta_dbl_tuple_;
    std::ostream* msgs_;

  public:
    using scalar_t = typename stan::return_type_t<Tt, T_init, T_par...>;
    using state_t = Eigen::Matrix<scalar_t, -1, 1>;
    static constexpr bool is_var_ts  = stan::is_var<Tt>::value;
    static constexpr bool is_var_y0  = stan::is_var<T_init>::value;
    static constexpr bool is_var_par = stan::is_var<stan::return_type_t<T_par...>>::value;
    static constexpr bool use_fwd_sens = is_var_y0 || is_var_par;

    const double t0_;
    const std::vector<Tt>& ts_;
    const Eigen::Matrix<T_init, -1, 1>& y0_;
    std::tuple<const T_par&...> theta_ref_tuple_;
    std::tuple<const Eigen::Matrix<T_init, -1, 1>&, const T_par&..., const std::vector<Tt>&> ode_arg_tuple_;
    std::tuple<decltype(stan::math::deep_copy_vars(std::declval<const T_par&>()))...> theta_local_tuple_;
    const size_t N;
    const size_t M;
    const size_t ns;
    const size_t system_size;
    std::vector<double> y0_fwd_system; // internally we use std::vector
    Eigen::VectorXd y_work, dydt_work, y_dbl_work, dydt_dbl_work, g_work;
    braid_data braid;

    PMXVariadicOdeSystem(const F& f,
                         double t0,
                         const std::vector<Tt>& ts,
                         const Eigen::Matrix<T_init, -1, 1>& y0,
                         std::ostream* msgs,
                         const T_par&... args)
      : f_(f),
        f_tuple_(f_),
        theta_dbl_tuple_(stan::math::value_of(args)...),
        msgs_(msgs),
        braid(),
        t0_(t0),
        ts_(ts),
        y0_(y0),
        theta_ref_tuple_(std::forward_as_tuple(args...)),
        ode_arg_tuple_(std::forward_as_tuple(y0_, args..., ts_)),
        theta_local_tuple_(stan::math::deep_copy_vars(args)...),
        N(y0.size()),
        M(stan::math::count_vars(args...)),
        ns((is_var_y0 ? N : 0) + M),
        system_size(N + N * ns),
        y0_fwd_system(system_size, 0.0),
        y_work(system_size),
        dydt_work(system_size),
        y_dbl_work(N),
        dydt_dbl_work(N),
        g_work(is_var_par? M : 0)
    {
      const char* caller = "PMX Variadic ODE System";
      torsten::dsolve::ode_check(y0_, t0_, ts_, caller, theta_ref_tuple_);

      // initial state
      for (size_t i = 0; i < N; ++i) {
        y0_fwd_system[i] = stan::math::value_of(y0.coeffRef(i));
      }
      if (is_var_y0)  {
        for (size_t i = 0; i < N; ++i) {
          y0_fwd_system[N + i * N + i] = 1.0;
        }
      }
    }

    static Eigen::VectorXd null_dbl_state(size_t n_size) {
      return Eigen::VectorXd::Zero(n_size);
    }

    static state_t null_state(size_t n_size) {
      return Eigen::Matrix<scalar_t, -1, 1>::Zero(n_size);
    }

    /*
     * retrieving a vector of vars that will be used as parameters
     */
    inline auto& vars() const {
      return ode_arg_tuple_;
    }

    /**
     * Evaluate RHS of the ODE(the combined system)
     * @param y current dependent value, arranged as {y, dy_dp1, dy_dp2...}
     * @param dy_dt ODE RHS to be filled.
     * @param t current indepedent value
     */
    inline void operator()(const std::vector<double> & y, std::vector<double> & dydt,
                           double t) {
      // dydt.resize(system_size);
      stan::math::check_size_match("PMXVariadicOdeSystem", "y", y.size(), "dy_dt", dydt.size());

      for (auto i = 0; i < system_size; ++i) {
        y_work.coeffRef(i) = y[i];
      }

      rhs_impl(y_work, dydt_work, t);

      for (auto i = 0; i < system_size; ++i) {
        dydt[i] = dydt_work.coeffRef(i);
      }
    }

    /*
     * evaluate RHS with data only inputs.
     */
    inline Eigen::VectorXd& dbl_rhs_impl(double t, const Eigen::VectorXd& y) {
      dydt_dbl_work = f_tuple_(t, y, msgs_, theta_dbl_tuple_);
      stan::math::check_size_match("PMXVariadicOdeSystem", "y", y.size(), "dy_dt", dydt_dbl_work.size());
      return dydt_dbl_work;
    }

    /**
     * evaluate RHS with data only inputs.
     */
    inline Eigen::VectorXd& dbl_rhs_impl(double t, const N_Vector& nv_y) {
      for (int i = 0; i < N; ++i) {
        y_dbl_work[i] = NV_Ith_S(nv_y, i);
      }
      return dbl_rhs_impl(t, y_dbl_work);
    }

    /**
     * evaluate RHS with data only inputs for N_Vector data
     */    
    inline void operator()(double t, N_Vector& nv_y, N_Vector& ydot) {
      Eigen::Map<Eigen::VectorXd>(NV_DATA_S(ydot), N) = dbl_rhs_impl(t, nv_y);
    }

    static int cvodes_rhs(double t, N_Vector y, N_Vector ydot, void* user_data) {
      Ode* ode = static_cast<Ode*>(user_data);
      (*ode)(t, y, ydot);
      return 0;
    }

    static int arkode_combined_rhs(double t, N_Vector y, N_Vector ydot, void* user_data) {
      Ode* ode = static_cast<Ode*>(user_data);
      for (auto i = 0; i < ode -> system_size; ++i) {
        ode -> y_work.coeffRef(i) = NV_Ith_S(y, i);
      }

      ode -> rhs_impl(ode -> y_work, ode -> dydt_work, t);

      for (auto i = 0; i < ode -> system_size; ++i) {
        NV_Ith_S(ydot, i) = ode -> dydt_work.coeffRef(i);
      }
      return 0;
    }

    // Equavalent to ARKODE' default WRMS
    static int wrms_fn1(N_Vector y, N_Vector ewt, void* user_data) {
      Ode* ode = static_cast<Ode*>(user_data);
      // dydt_work contains the latest evaluation the combined RHS
      // we use it as proxy to the current RHS.
      for (int i = 0; i < ode -> system_size; ++i) {
        NV_Ith_S(ewt, i) = 1.0/(ode -> atol + ode -> rtol * abs(NV_Ith_S(y, i)));
      }

      return 0;
    }

    // Alternative ARKODE' WRMS norm weight, we can use odeint's
    // weight norm that incorporates derivatives
    static int wrms_fn2(N_Vector y, N_Vector ewt, void* user_data) {
      Ode* ode = static_cast<Ode*>(user_data);
      // dydt_work contains the latest evaluation the combined RHS
      // we use it as proxy to the current RHS.
      double dt;
      ERKStepGetLastStep(ode -> mem_ptr, &dt);
      for (int i = 0; i < ode -> system_size; ++i) {
        NV_Ith_S(ewt, i) = 1.0/(ode -> atol + ode -> rtol * abs(NV_Ith_S(y, i)));
      }
      return 0;
    }

    // Alternative ARKODE' WRMS norm weight, we can use odeint's
    // weight norm that incorporates derivatives
    static int wrms_fn3(N_Vector y, N_Vector ewt, void* user_data) {
      Ode* ode = static_cast<Ode*>(user_data);
      // dydt_work contains the latest evaluation the combined RHS
      // we use it as proxy to the current RHS.
      double dt;
      ERKStepGetLastStep(ode -> mem_ptr, &dt);
      for (int i = 0; i < ode -> system_size; ++i) {
        NV_Ith_S(ewt, i) = 1.0/(ode -> atol + ode -> rtol * std::max(abs(NV_Ith_S(y, i)),
                                                                     dt * abs(ode -> dydt_work[i])));
      }
      return 0;
    }

    static int arkode_erk_adapt(N_Vector y, double t, double h1, double h2, double h3, double e1, double e2, double e3, int q, int p, double* hnew, void* user_data)
    {
      // Ode* ode = static_cast<Ode*>(user_data);
      *hnew = h1;
      if (e1 > 1) {
        *hnew = h1 * std::max( 0.9 * std::pow(e1 , -0.25) , 0.2 );
      } else if (e1 < 0.5) {
        *hnew = h1 * std::min( 0.9 * std::pow(e1 , -0.20) , 5.0 );
      }
      return 0;
    }

    /**
     * evalute RHS of the entire system, possibly including
     * the forward sensitivity equation components in @c y and @c dy_dt.
     */
    void rhs_impl(const Eigen::VectorXd & y, Eigen::VectorXd & dydt, double t) {
      if (!(is_var_y0 || is_var_par)) {
        dydt = dbl_rhs_impl(t, y);
        return;
      }

      dydt.fill(0.0);
      stan::math::nested_rev_autodiff nested;

      stan::math::vector_v yv(N);
      for (size_t i = 0; i < N; ++i) { yv.coeffRef(i) = y.coeffRef(i); }
      stan::math::vector_v fyv(f_tuple_(t, yv, msgs_, theta_local_tuple_));
      stan::math::check_size_match("PMXVariadicOdeSystem", "y", yv.size(), "dy_dt", fyv.size());

      Eigen::VectorXd& g = g_work;
      for (size_t i = 0; i < N; ++i) {
        if (i > 0) {
          nested.set_zero_all_adjoints();            
        }
        dydt.coeffRef(i) = (fyv.coeffRef(i)).val();
        (fyv.coeffRef(i)).grad();

        // df/dy*s_i term, for i = 1...ns
        for (size_t j = 0; j < ns; ++j) {
          for (size_t k = 0; k < N; ++k) {
            dydt.coeffRef(N + N * j + i) += y.coeffRef(N + N * j + k) * (yv.coeffRef(k)).adj();
          }
        }

        // df/dp_i term, for i = n...n+m-1
        if (is_var_par) {
          memset(g.data(), 0, sizeof(double) * g.size());
          stan::math::apply([&](auto&&... args) {stan::math::accumulate_adjoints(g.data(), args...);},
                theta_local_tuple_);
          for (size_t j = 0; j < M; ++j) {
            dydt.coeffRef(N + N * (ns - M + j) + i) += g.coeffRef(j);
          }
        }

        stan::math::for_each([](auto&& arg) { stan::math::zero_adjoints(arg); }, theta_local_tuple_);
      }
    }

    /**
     * Calculate sensitivity rhs using CVODES vectors. The
     * internal workspace is allocated by @c CvodesService.
     */
    inline void operator()(int ns, double t, N_Vector nv_y, N_Vector ydot,
                           N_Vector* ys, N_Vector* ysdot,
                           N_Vector temp1, N_Vector temp2) {
      using stan::math::var;
      using stan::math::vector_v;
      using stan::math::vector_d;

      // initialize ysdot
      for (int i = 0; i < ns; ++i) N_VConst(0.0, ysdot[i]);

      stan::math::nested_rev_autodiff nested;

      auto local_theta_tuple_ = deep_copy_tuple()(theta_ref_tuple_);

      vector_v yv(N);
      for (size_t i = 0; i < N; ++i) { yv[i] = NV_Ith_S(nv_y, i); }
      vector_v dydt(f_tuple_(t, yv, msgs_, local_theta_tuple_));

      stan::math::check_size_match("PMXOdeSystem", "dydt", dydt.size(), "states", N);

      Eigen::VectorXd g(M);
      for (int j = 0; j < N; ++j) {
        if (j > 0) {
          nested.set_zero_all_adjoints();
        }
        dydt[j].grad();

        // df/dy*s_i term, for i = 1...ns
        for (int i = 0; i < ns; ++i) {
          auto ysp = N_VGetArrayPointer(ys[i]);
          auto nvp = N_VGetArrayPointer(ysdot[i]);
          for (int k = 0; k < N; ++k) {
            nvp[j] += yv[k].adj() * ysp[k];              
          }
        }

        // df/dp_i term, for i = n...n+m-1
        if (is_var_par) {
          g.fill(0);
          stan::math::apply([&](auto&&... args) {stan::math::accumulate_adjoints(g.data(), args...);},
                local_theta_tuple_);
          for (int i = 0; i < M; ++i) {
            auto nvp = N_VGetArrayPointer(ysdot[ns - M + i]);
            nvp[j] += g[i];
          }
        }

        stan::math::for_each([](auto&& arg) { stan::math::zero_adjoints(arg); }, local_theta_tuple_);
      }
    }

    static int cvodes_sens_rhs(int ns, double t, N_Vector y, N_Vector ydot,
                               N_Vector* ys, N_Vector* ysdot, void* user_data,
                               N_Vector temp1, N_Vector temp2) {
      if (!use_fwd_sens) return 0;
      Ode* ode = static_cast<Ode*>(user_data);
      (*ode)(ns, t, y, ydot, ys, ysdot, temp1, temp2);
      return 0;
    }

    /**
     * return a closure for CVODES residual callback using a
     * non-capture lambda.
     *
     * @tparam Ode type of Ode
     * @return RHS function for Cvodes
     */
    inline CVLsJacFn cvodes_jac() {
      return [](realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void* user_data, // NOLINT
                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) -> int {
        Ode* ode = static_cast<Ode*>(user_data);
        ode -> jac(t, y, fy, J);
        return 0;
      };
    }

    /**
     * evaluate Jacobian matrix using current state, store
     * the result in @c SUNMatrix J.
     *
     * @param t current time
     * @param y current y
     * @param fy current f(y)
     * @param J Jacobian matrix J(i,j) = df_i/dy_j
     */
    inline void jac(double t, N_Vector& nv_y, N_Vector& fy, SUNMatrix& J) {
      using stan::math::vector_v;

      stan::math::nested_rev_autodiff nested;

      auto local_theta_tuple_ = deep_copy_tuple()(theta_ref_tuple_);

      vector_v yv(N);
      for (size_t i = 0; i < N; ++i) { yv[i] = NV_Ith_S(nv_y, i); }
      vector_v dydt(f_tuple_(t, yv, msgs_, local_theta_tuple_));

      for (int i = 0; i < N; ++i) {
        nested.set_zero_all_adjoints();
        dydt[i].grad();
        for (int j = 0; j < N; ++j) {
          SM_ELEMENT_D(J, i, j) = yv[j].adj();
        }
      }
    }

#ifdef TORSTEN_BRAID
    static int braid_init(braid_App app, double t, braid_Vector *u_ptr) {
      int      flag;
      void     *user_data;

      // Get user data pointer
      ARKBraid_GetUserData(app, &user_data);
      Ode *ode = static_cast<Ode*>(user_data);

      // Create new vector
      N_Vector y = N_VNew_Serial(ode -> system_size);
      flag = SUNBraidVector_New(y, u_ptr);
      if (flag != 0) return 1;

      // Set initial condition at all time points
      if (t == ode -> t0_) {
        for (auto i = 0; i < ode -> system_size; ++i) {
          NV_Ith_S(y, i) = ode -> y0_fwd_system[i];
        }
      } else {
        N_VConst(0.0, y);
      }

      return 0;        
    }

    static int braid_access(braid_App app, braid_Vector u, braid_AccessStatus astatus) {
      int       flag;    // return flag
      int       iter;    // current iteration number
      int       level;   // current level
      int       done;    // has XBraid finished
      int       index;   // time point index
      double    t;       // current time
      void     *user_data;
      Ode      *ode;

      // Timing variables
      std::chrono::time_point<std::chrono::steady_clock> t1;
      std::chrono::time_point<std::chrono::steady_clock> t2;

      // Start timer
      t1 = std::chrono::steady_clock::now();

      // Get user data pointer
      ARKBraid_GetUserData(app, &user_data);
      ode = static_cast<Ode*>(user_data);

      // Get current time, iteration, level, and status
      braid_AccessStatusGetTILD(astatus, &t, &iter, &level, &done);
      braid_AccessStatusGetTIndex(astatus, &index);

      // Output on fine level when XBraid has finished
      if (level == 0 && done) {
        // Get current time index and number of fine grid points
        int index;
        int ntpts;
        braid_AccessStatusGetTIndex(astatus, &index);
        braid_AccessStatusGetNTPoints(astatus, &ntpts);

        // Extract NVector
        N_Vector y = NULL;
        SUNBraidVector_GetNVector(u, &y);

        // Write visualization files
        // if (udata->output == 2)
        //   {
        //     // Get output frequency (ensure the final time is output)
        //     int qout = ntpts / udata->nout;
        //     int rout = ntpts % udata->nout;
        //     int nout = (rout > 0) ? udata->nout + 2 : udata->nout + 1;

        //     // Output problem information
        //     if (index == 0)
        //       {
        //         ofstream dout;
        //         dout.open("heat2d_info.txt");
        //         dout <<  "xu  " << udata->xu << endl;
        //         dout <<  "yu  " << udata->yu << endl;
        //         dout <<  "nx  " << udata->nx << endl;
        //         dout <<  "ny  " << udata->ny << endl;
        //         dout <<  "nt  " << nout      << endl;
        //         dout.close();
        //       }

        //     // Output solution and error
        //     if (!(index % qout) || index == ntpts)
        //       {
        //         // Open output streams
        //         stringstream fname;
        //         fname << "heat2d_solution."
        //               << setfill('0') << setw(6) << index / qout << ".txt";

        //         udata->uout.open(fname.str());
        //         udata->uout << scientific;
        //         udata->uout << setprecision(numeric_limits<realtype>::digits10);

        //         fname.str("");
        //         fname.clear();
        //         fname << "heat2d_error."
        //               << setfill('0') << setw(6) << index / qout << ".txt";

        //         udata->eout.open(fname.str());
        //         udata->eout << scientific;
        //         udata->eout << setprecision(numeric_limits<realtype>::digits10);

        //         // Compute the error
        //         flag = SolutionError(t, y, udata->e, udata);
        //         if (check_flag(&flag, "SolutionError", 1)) return 1;

        //         // Output solution to disk
        //         realtype *yarray = N_VGetArrayPointer(y);
        //         if (check_flag((void *) yarray, "N_VGetArrayPointer", 0)) return -1;

        //         udata->uout << t << " ";
        //         for (sunindextype i = 0; i < udata->nodes; i++)
        //           {
        //             udata->uout << yarray[i] << " ";
        //           }
        //         udata->uout << endl;

        //         // Output error to disk
        //         realtype *earray = N_VGetArrayPointer(udata->e);
        //         if (check_flag((void *) earray, "N_VGetArrayPointer", 0)) return -1;

        //         udata->eout << t << " ";
        //         for (sunindextype i = 0; i < udata->nodes; i++)
        //           {
        //             udata->eout << earray[i] << " ";
        //           }
        //         udata->eout << endl;

        //         // Close output streams
        //         udata->uout.close();
        //         udata->eout.close();
        //       }
        //   }

        // Output final error
        // if (index == ntpts) {
        //   // Compute the max error
        //   flag = SolutionError(t, y, udata->e, udata);
        //   if (check_flag(&flag, "SolutionError", 1)) return 1;

        //   realtype maxerr = N_VMaxNorm(udata->e);

        //   cout << scientific;
        //   cout << setprecision(numeric_limits<realtype>::digits10);
        //   cout << "  Max error = " << maxerr << endl << endl;
        // }
      }

      // Stop timer
      t2 = std::chrono::steady_clock::now();

      // Update timing
      // ode.braid.accesstime += std::chrono::duration<double>(t2 - t1).count();

      return 0;
    }
#endif
  };
}  // namespace dsolve
}  // namespace torsten
#endif
