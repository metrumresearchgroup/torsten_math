#ifndef STAN_MATH_TEST_FIXTURE_ODE_STIFF_B1_HPP
#define STAN_MATH_TEST_FIXTURE_ODE_STIFF_B1_HPP

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
struct stiff_b1_base {
  struct stiff_b1_rhs {
    template <typename T0, typename T1, typename T2>
    Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> operator()(const T0& t_in, const T1& y_in, std::ostream* msgs,
                                                                 const T2& theta) const {
      Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> res(4);
      auto& a = theta[0];
      auto& b = theta[1];
      auto& c = theta[2];
      auto& d = theta[3];
      res(0) = -y_in(0) + y_in(1);
      res(1) = -a * y_in(0) - y_in(1);
      res(2) = -b * y_in(2) + y_in(3);
      res(3) = -c * y_in(2) - d * y_in(3);
      return res;
    }
  };

  using T_init = std::tuple_element_t<2, T>;
  using T_param = std::tuple_element_t<3, T>;

  stan::math::nested_rev_autodiff nested;
  stiff_b1_rhs f;
  std::tuple_element_t<0, T> s1;
  std::tuple_element_t<1, T> s2;
  Eigen::Matrix<T_init, -1, 1> y0;
  std::vector<T_param> theta;
  double t0;
  std::vector<double> ts;
  double rtol;
  double atol;
  int max_num_step;

  stiff_b1_base() : nested(),
                    f(), s1(), s2(),
                    y0(4),
                    theta{100, 100, 100000, 100},
                    t0(0.0),
                    ts{100},
                    rtol(1.e-6),
                    atol(1.e-6),
                    max_num_step(100000) {
    y0(0) = 1.0;
    y0(1) = 0.0;
    y0(2) = 1.0;
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
struct stiff_b1_benchmark_tolerance_1_test : public stiff_b1_base<T>,
                                             public ODEBenchmarkFixture<stiff_b1_benchmark_tolerance_1_test<T>> {
  stiff_b1_benchmark_tolerance_1_test() {
    this -> rtol = 1.e-3;
    this -> atol = 1.e-3;
  }
};

template <typename T>
struct stiff_b1_benchmark_tolerance_2_test : public stiff_b1_base<T>,
                                             public ODEBenchmarkFixture<stiff_b1_benchmark_tolerance_2_test<T>> {
  stiff_b1_benchmark_tolerance_2_test() {
    this -> rtol = 1.e-6;
    this -> atol = 1.e-6;
  }
};

template <typename T>
struct stiff_b1_benchmark_tolerance_3_test : public stiff_b1_base<T>,
                                             public ODEBenchmarkFixture<stiff_b1_benchmark_tolerance_3_test<T>> {
  stiff_b1_benchmark_tolerance_3_test() {
    this -> rtol = 1.e-8;
    this -> atol = 1.e-8;
  }
};

template <typename T>
struct stiff_b1_benchmark_tolerance_4_test : public stiff_b1_base<T>,
                                             public ODEBenchmarkFixture<stiff_b1_benchmark_tolerance_4_test<T>> {
  stiff_b1_benchmark_tolerance_4_test() {
    this -> rtol = 1.e-12;
    this -> atol = 1.e-12;
  }
};

#endif
