#ifndef TEST_UNIT_MATH_REV_FUNCTOR_ODE_BENCHMARK_FIXTURE_HPP
#define TEST_UNIT_MATH_REV_FUNCTOR_ODE_BENCHMARK_FIXTURE_HPP

#include <stan/math/rev.hpp>
#include <test/prob/utility.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <type_traits>
#include <stan/math/torsten/test/unit/test_util.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <string>


template <class ode_problem_type>
struct ODEBenchmarkFixture : public ::testing::Test {
  /**
   * test ODE solution from one solver against another
   *
   * Require function in ode_type:
   * - Matrix<T, -1, 1> apply_solver_ctrl(solver)
   * - solver1(): return ode integrator of type 1
   * - solver2(): return ode integrator of type 2
   * - theta: parameter vector
   *
   */
  void test_compare_ctrl(double val_tol, double adj_tol) {
    ode_problem_type& ode = static_cast<ode_problem_type&>(*this);
    test_compare_ctrl_impl(ode, val_tol, adj_tol);
  }

  template<typename Ode, stan::require_var_t<typename Ode::T_init>* = nullptr,
    stan::require_var_t<typename Ode::T_param>* = nullptr>
  void test_compare_ctrl_impl(Ode& ode, double val_tol, double adj_tol) {
    auto sol1 = ode.apply_solver_ctrl(ode.solver1());
    auto sol2 = ode.apply_solver_ctrl(ode.solver2());
    EXPECT_ARRAY2D_VAL_NEAR(sol1, sol2, val_tol);
    EXPECT_ARRAY2D_ADJ_NEAR(sol1, sol2, ode.y0, ode.nested, adj_tol, "benchmark");
    EXPECT_ARRAY2D_ADJ_NEAR(sol1, sol2, ode.theta, ode.nested, adj_tol, "benchmark");
  }

  template<typename Ode, stan::require_var_t<typename Ode::T_init>* = nullptr,
    stan::require_not_var_t<typename Ode::T_param>* = nullptr>
  void test_compare_ctrl_impl(Ode& ode, double val_tol, double adj_tol) {
    auto sol1 = ode.apply_solver_ctrl(ode.solver1());
    auto sol2 = ode.apply_solver_ctrl(ode.solver2());
    EXPECT_ARRAY2D_VAL_NEAR(sol1, sol2, val_tol);
    EXPECT_ARRAY2D_ADJ_NEAR(sol1, sol2, ode.y0, ode.nested, adj_tol, "benchmark");
  }

  template<typename Ode, stan::require_not_var_t<typename Ode::T_init>* = nullptr,
    stan::require_var_t<typename Ode::T_param>* = nullptr>
  void test_compare_ctrl_impl(Ode& ode, double val_tol, double adj_tol) {
    auto sol1 = ode.apply_solver_ctrl(ode.solver1());
    auto sol2 = ode.apply_solver_ctrl(ode.solver2());
    EXPECT_ARRAY2D_VAL_NEAR(sol1, sol2, val_tol);
    EXPECT_ARRAY2D_ADJ_NEAR(sol1, sol2, ode.theta, ode.nested, adj_tol, "benchmark");
  }

  template<typename Ode, stan::require_all_not_var_t<typename Ode::T_init, typename Ode::T_param>* = nullptr>
  void test_compare_ctrl_impl(Ode& ode, double val_tol, double adj_tol) {
    auto sol1 = ode.apply_solver_ctrl(ode.solver1());
    auto sol2 = ode.apply_solver_ctrl(ode.solver2());
    EXPECT_ARRAY2D_VAL_NEAR(sol1, sol2, val_tol);
  }
};
#endif
