#ifndef STAN_MATH_TEST_FIXTURE_ODE_STIFF_E1_HPP
#define STAN_MATH_TEST_FIXTURE_ODE_STIFF_E1_HPP

// W. ENRIGHT, T. HULL AND B. LINDBERG, Comparing numerical methods for stiff systems of 
// ordinary differential equations, BIT, 15 (1975), pp. 1

#include <stan/math/rev.hpp>
#include <boost/numeric/odeint.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/functor/test_fixture_ode.hpp>
#include <stan/math/torsten/test/unit/dsolve/test_fixture_ode_benchmark.hpp>
#include <test/unit/util.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <string>

template <typename T>
struct stiff_e1_base {
  struct stiff_e1_rhs {
    template <typename T0, typename T1, typename T2>
    Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> operator()(const T0& t_in, const T1& y, std::ostream* msgs,
                                                                 const T2& theta) const {
      Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> res(4);
      auto& gamma = theta[0];
      auto gamma_2 = gamma * gamma;
      auto gamma_3 = gamma_2 * gamma;
      auto gamma_4 = gamma_2 * gamma_2;
      res(0) = y(1);
      res(1) = y(2);
      res(2) = y(3);
      res(3) = (y(0) * y(0) - sin(y(0)) - gamma_4) * y(0) + 
                (y(1) * y(2) / (y(0) * y(0) + 1) - 4 * gamma_3) * y(1) + 
                (1 - 6 * gamma * gamma) * y(2) +
                (10 * stan::math::exp(-y(3) * y(3)) - 4 * gamma) * y(3) + 1;
      return res;
    }
  };

  using T_init = std::tuple_element_t<2, T>;
  using T_param = std::tuple_element_t<3, T>;

  stan::math::nested_rev_autodiff nested;
  stiff_e1_rhs f;
  std::tuple_element_t<0, T> s1;
  std::tuple_element_t<1, T> s2;
  Eigen::Matrix<T_init, -1, 1> y0;
  std::vector<T_param> theta;
  double t0;
  std::vector<double> ts;
  double rtol;
  double atol;
  int max_num_step;

  stiff_e1_base() : nested(),
                    f(), s1(), s2(),
                    y0(4),
                    theta{100},
                    t0(0),
                    ts{10},
                    rtol(1.e-6),
                    atol(1.e-6),
                    max_num_step(100000) {
    y0(0) = 0.0;
    y0(1) = 0.0;
    y0(2) = 0.0;
    y0(3) = 0.0;
  }

  std::vector<double>& times() { return ts; }
  Eigen::Matrix<T_init, -1, 1>& init() { return y0; }
  std::vector<T_param>& param() { return theta; }

  std::tuple_element_t<0, T>& solver1() { return s1; }
  std::tuple_element_t<1, T>& solver2() { return s2; }

  template<typename S>
  std::vector<Eigen::Matrix<stan::return_type_t<T_init, T_param>, -1, 1>>
  apply_solver_ctrl(S& solver) {
    return solver(this->f, this -> y0, this->t0, this->ts, this->rtol, this->atol,
                  this->max_num_step, nullptr, this -> theta);
  }
};

template <typename T>
struct stiff_e1_benchmark_tolerance_1_test : public stiff_e1_base<T>,
                                             public ODEBenchmarkFixture<stiff_e1_benchmark_tolerance_1_test<T>> {
  stiff_e1_benchmark_tolerance_1_test() {
    this -> rtol = 1.e-3;
    this -> atol = 1.e-3;
  }
};

template <typename T>
struct stiff_e1_benchmark_tolerance_2_test : public stiff_e1_base<T>,
                                             public ODEBenchmarkFixture<stiff_e1_benchmark_tolerance_2_test<T>> {
  stiff_e1_benchmark_tolerance_2_test() {
    this -> rtol = 1.e-6;
    this -> atol = 1.e-6;
  }
};

template <typename T>
struct stiff_e1_benchmark_tolerance_3_test : public stiff_e1_base<T>,
                                             public ODEBenchmarkFixture<stiff_e1_benchmark_tolerance_3_test<T>> {
  stiff_e1_benchmark_tolerance_3_test() {
    this -> rtol = 1.e-8;
    this -> atol = 1.e-8;
  }
};

template <typename T>
struct stiff_e1_benchmark_tolerance_4_test : public stiff_e1_base<T>,
                                             public ODEBenchmarkFixture<stiff_e1_benchmark_tolerance_4_test<T>> {
  stiff_e1_benchmark_tolerance_4_test() {
    this -> rtol = 1.e-12;
    this -> atol = 1.e-12;
  }
};

#endif
