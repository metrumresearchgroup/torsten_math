#ifndef TEST_UNIT_TORSTEN_TEST_FIXTURE_MODEL
#define TEST_UNIT_TORSTEN_TEST_FIXTURE_MODEL

#include <gtest/gtest.h>
#include <stan/math/rev/fun/fmax.hpp>
#include <boost/numeric/odeint.hpp>
#include <stan/math/torsten/test/unit/test_macros.hpp>
#include <stan/math/torsten/test/unit/test_functors.hpp>
#include <stan/math/torsten/test/unit/test_util.hpp>
#include <test/unit/math/prim/functor/harmonic_oscillator.hpp>
#include <test/unit/math/prim/functor/lorenz.hpp>
#include <nvector/nvector_serial.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

template<typename T>
struct TorstenPMXTestBase;

template<template<typename> class child_type, typename T>
struct TorstenPMXTestBase<child_type<T> > : public testing::Test {
  /// solver type for analytical solutions
  using sol1_t = std::tuple_element_t<0, T>;
  /// solver type for numerical ode solutions
  using sol2_t = std::tuple_element_t<1, T>;
  /// time type
  using time_t = std::tuple_element_t<2, T>;
  /// amount type
  using amt_t = std::tuple_element_t<3, T>;
  /// rate type
  using rate_t = std::tuple_element_t<4, T>;
  /// II type
  using ii_t = std::tuple_element_t<5, T>;
  /// param type
  using param_t = std::tuple_element_t<6, T>;
  /// F type
  using biovar_t = std::tuple_element_t<7, T>;
  /// lag time type
  using tlag_t = std::tuple_element_t<8, T>;
  // ODE functor type
  using ode_t = std::tuple_element_t<9, T>;

  stan::math::nested_rev_autodiff nested;
  int nt;
  std::vector<time_t> time;
  std::vector<amt_t> amt;
  std::vector<rate_t> rate;
  std::vector<int> cmt;
  std::vector<int> evid;
  std::vector<ii_t> ii;
  std::vector<int> addl;
  std::vector<int> ss;
  std::vector<std::vector<param_t> > theta;
  std::vector<Eigen::Matrix<param_t, -1, -1> > pMatrix;
  std::vector<std::vector<biovar_t> > biovar;
  std::vector<std::vector<tlag_t> > tlag;

  // solvers
  sol1_t sol1;
  sol2_t sol2;

  // for ODE integrator
  int ncmt;
  double t0;
  std::vector<double> x_r;
  std::vector<int> x_i;
  double rtol;
  double atol;
  int max_num_steps;
  double as_rtol;
  double as_atol;
  int as_max_num_steps;
  std::ostream* msgs;

  TorstenPMXTestBase() : 
    t0(0.0),
    rtol             {1.E-10},
    atol             {1.E-10},
    max_num_steps    {100000},
    as_rtol          {1.E-4},
    as_atol          {1.E-6},
    as_max_num_steps {100},
    msgs             {nullptr}
  {
    // nested.set_zero_all_adjoints();    
  }

  void resize(int n) {
    nt = n;
    time.resize(nt);
    amt .resize(nt);
    rate.resize(nt);
    cmt .resize(nt);
    evid.resize(nt);
    ii  .resize(nt);
    addl.resize(nt);
    ss  .resize(nt);

    std::fill(time.begin(), time.end(), 0);
    std::fill(amt .begin(), amt .end(), 0);
    std::fill(rate.begin(), rate.end(), 0);
    std::fill(cmt .begin(), cmt .end(), ncmt);
    std::fill(evid.begin(), evid.end(), 0);
    std::fill(ii  .begin(), ii  .end(), 0);
    std::fill(addl.begin(), addl.end(), 0);
    std::fill(ss  .begin(), ss  .end(), 0);
  }

  void compare_solvers_val() {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = sol1(time, amt, rate, ii, evid, cmt, addl, ss, theta, biovar, tlag);
    auto res2 = fixture.solver2_solution();
    EXPECT_MAT_VAL_FLOAT_EQ(res1, res2);
  }

  void compare_solvers_val(double tol) {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = sol1(time, amt, rate, ii, evid, cmt, addl, ss, theta, biovar, tlag);
    auto res2 = fixture.solver2_solution();
    EXPECT_MAT_VAL_NEAR(res1, res2, tol);
  }

  void compare_solvers_adj(const std::vector<double>& p, double tol, const char*) {}

  void compare_solvers_adj(std::vector<stan::math::var>& p, double tol,
                           const char* diagnostic_msg) {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = sol1(time, amt, rate, ii, evid, cmt, addl, ss, theta, biovar, tlag);
    auto res2 = fixture.solver2_solution();
    EXPECT_MAT_ADJ_NEAR(res1, res2, p, this -> nested, tol, diagnostic_msg);
  }
};

/** 
 * Default test fixture assuming numerical ODE solver for 
 * <code>sol2_t</code>.
 */
template<typename T>
struct TorstenPMXTest : public TorstenPMXTestBase<TorstenPMXTest<T> >
{
  auto solver2_solution () {
    typename TorstenPMXTestBase<TorstenPMXTest<T> >::ode_t f;
    return this -> sol2(f, this -> ncmt, this -> time, this -> amt, this -> rate, this -> ii, this -> evid, this -> cmt, this -> addl, this ->
                        ss, this -> theta, this -> biovar, this -> tlag, this -> rtol, this -> atol, this -> max_num_steps, this -> as_rtol, this -> as_atol, this -> as_max_num_steps,
                        nullptr);
  }
};

/** 
 * Specialization of test fixture assuming linear ODE solver for 
 * <code>sol2_t</code>.
 */
template<typename sol1_type, typename... Ts>
struct TorstenPMXTest<std::tuple<sol1_type, pmx_solve_linode_functor, Ts...>> :
  public TorstenPMXTestBase<TorstenPMXTest<std::tuple<sol1_type, pmx_solve_linode_functor, Ts...>>>
{
  auto solver2_solution () {
    return this -> sol2(this -> time, this -> amt, this -> rate, this -> ii, this -> evid, this -> cmt, this -> addl, this -> ss, this -> pMatrix, this -> biovar, this -> tlag);
  }
};

#endif

