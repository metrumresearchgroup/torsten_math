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


/** 
 * Helper functor to apply solver. Default solver assumes numerical ODE for 
 * <code>solver_func_t</code>.
 */
template<typename solver_func_t>
struct apply_solver {
  solver_func_t const& s;
  apply_solver(solver_func_t const& sol) : s(sol) {}

  template<typename child_test_t>
  auto operator()(child_test_t* test_ptr) {
  typename child_test_t::ode_t f;
  return s(f, test_ptr -> ncmt, test_ptr -> time, test_ptr -> amt, test_ptr -> rate, test_ptr -> ii, test_ptr -> evid, test_ptr -> cmt, test_ptr -> addl, test_ptr -> ss,
           test_ptr -> theta, test_ptr -> biovar, test_ptr -> tlag,
           test_ptr -> rtol, test_ptr -> atol, test_ptr -> max_num_steps,
           test_ptr -> as_rtol, test_ptr -> as_atol, test_ptr -> as_max_num_steps,
           nullptr);
  }
};

template<>
struct apply_solver<pmx_solve_onecpt_functor> {
  pmx_solve_onecpt_functor const& s;
  apply_solver(pmx_solve_onecpt_functor const& sol) : s(sol) {}

  template<typename child_test_t>
  auto operator()(child_test_t* test_ptr) {
  return s(test_ptr -> time, test_ptr -> amt, test_ptr -> rate, test_ptr -> ii, test_ptr -> evid, test_ptr -> cmt, test_ptr -> addl, test_ptr -> ss,
           test_ptr -> theta, test_ptr -> biovar, test_ptr -> tlag);
  }
};

template<>
struct apply_solver<pmx_solve_twocpt_functor> {
  pmx_solve_twocpt_functor const& s;
  apply_solver(pmx_solve_twocpt_functor const& sol) : s(sol) {}

  template<typename child_test_t>
  auto operator()(child_test_t* test_ptr) {
  return s(test_ptr -> time, test_ptr -> amt, test_ptr -> rate, test_ptr -> ii, test_ptr -> evid, test_ptr -> cmt, test_ptr -> addl, test_ptr -> ss,
           test_ptr -> theta, test_ptr -> biovar, test_ptr -> tlag);
  }
};

template<>
struct apply_solver<pmx_solve_onecpt_effcpt_functor> {
  pmx_solve_onecpt_effcpt_functor const& s;
  apply_solver(pmx_solve_onecpt_effcpt_functor const& sol) : s(sol) {}

  template<typename child_test_t>
  auto operator()(child_test_t* test_ptr) {
  return s(test_ptr -> time, test_ptr -> amt, test_ptr -> rate, test_ptr -> ii, test_ptr -> evid, test_ptr -> cmt, test_ptr -> addl, test_ptr -> ss,
           test_ptr -> theta, test_ptr -> biovar, test_ptr -> tlag);
  }
};

template<>
struct apply_solver<pmx_solve_linode_functor> {
  pmx_solve_linode_functor const& s;
  apply_solver(pmx_solve_linode_functor const& sol) : s(sol) {}

  template<typename child_test_t>
  auto operator()(child_test_t* test_ptr) {
    return s(test_ptr -> time, test_ptr -> amt, test_ptr -> rate, test_ptr -> ii, test_ptr -> evid,
             test_ptr -> cmt, test_ptr -> addl, test_ptr -> ss, test_ptr -> pMatrix,
             test_ptr -> biovar, test_ptr -> tlag);
  }
};

template<typename T>
struct TorstenPMXTest;

template<template<typename> class child_type, typename T>
struct TorstenPMXTest<child_type<T> > : public testing::Test {
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

  TorstenPMXTest() : 
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

  void reset_events(int n) {
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

  void compare_val(Eigen::MatrixXd const& x) {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = apply_solver<sol1_t>(sol1)(&fixture);
    EXPECT_MAT_VAL_FLOAT_EQ(res1, x);
  }

  void compare_val(Eigen::MatrixXd const& x, double tol) {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = apply_solver<sol1_t>(sol1)(&fixture);
    EXPECT_MAT_VAL_NEAR(res1, x, tol);
  }

  void compare_solvers_val() {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = apply_solver<sol1_t>(sol1)(&fixture);
    auto res2 = apply_solver<sol2_t>(sol2)(&fixture);
    EXPECT_MAT_VAL_FLOAT_EQ(res1, res2);
  }

  void compare_solvers_val(double tol) {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = apply_solver<sol1_t>(sol1)(&fixture);
    auto res2 = apply_solver<sol2_t>(sol2)(&fixture);
    EXPECT_MAT_VAL_NEAR(res1, res2, tol);
  }

  void compare_solvers_adj(const std::vector<double>& p, double tol, const char*) {}

  void compare_solvers_adj(std::vector<stan::math::var>& p, double tol,
                           const char* diagnostic_msg) {
    child_type<T>& fixture = static_cast<child_type<T>&>(*this);
    auto res1 = apply_solver<sol1_t>(sol1)(&fixture);
    auto res2 = apply_solver<sol2_t>(sol2)(&fixture);
    EXPECT_MAT_ADJ_NEAR(res1, res2, p, this -> nested, tol, diagnostic_msg);
  }

  template<typename x_type>
  auto test_func_time(std::vector<x_type> const& x) {
    return sol1(x, amt, rate, ii, evid, cmt, addl, ss, theta, biovar, tlag);
  }

  template<typename x_type>
  auto test_func_amt(std::vector<x_type> const& x) {
    return sol1(time, x, rate, ii, evid, cmt, addl, ss, theta, biovar, tlag);
  }

  template<typename x_type>
  auto test_func_rate(std::vector<x_type> const& x) {
    return sol1(time, amt, x, ii, evid, cmt, addl, ss, theta, biovar, tlag);
  }

  template<typename x_type>
  auto test_func_ii(std::vector<x_type> const& x) {
    return sol1(time, amt, rate, x, evid, cmt, addl, ss, theta, biovar, tlag);
  }

  template<typename x_type>
  auto test_func_theta(std::vector<x_type> const& x) {
    std::vector<std::vector<x_type> > x_{x};
    return sol1(time, amt, rate, ii, evid, cmt, addl, ss, x_, biovar, tlag);
  }

  template<typename x_type>
  auto test_func_biovar(std::vector<x_type> const& x) {
    std::vector<std::vector<x_type> > x_{x};
    return sol1(time, amt, rate, ii, evid, cmt, addl, ss, theta, x_, tlag);
  }

  template<typename x_type>
  auto test_func_tlag(std::vector<x_type> const& x) {
    std::vector<std::vector<x_type> > x_{x};
    return sol1(time, amt, rate, ii, evid, cmt, addl, ss, theta, biovar, x_);
  }

#define ADD_FD_TEST(NAME, ARG_VEC)                                      \
  void test_finite_diff_##NAME(double h, double tol) {                  \
    EXPECT_MAT_FUNC_POSITIVE_PARAM_NEAR_FD(test_func_##NAME, ARG_VEC, nested, h, tol, #NAME); \
  }

  ADD_FD_TEST(amt, amt);
  ADD_FD_TEST(time, time);
  ADD_FD_TEST(rate, rate);
  ADD_FD_TEST(ii, ii);
  ADD_FD_TEST(theta, theta[0]);
  ADD_FD_TEST(biovar, biovar[0]);
  ADD_FD_TEST(tlag, tlag[0]);

#undef ADD_FD_TEST
};

#endif

