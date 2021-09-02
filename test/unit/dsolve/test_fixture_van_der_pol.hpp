#ifndef STAN_MATH_TEST_FIXTURE_ODE_VAN_DER_POL_HPP
#define STAN_MATH_TEST_FIXTURE_ODE_VAN_DER_POL_HPP

// J.M.  France*, I.  Gbmez, L.  RBndez, SDIRK methods for stiff ODES with oscillating solutions 
// Journal of Computational and Applied Mathematics 81(1997) 197-209

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
struct van_der_pol_base {
  struct van_der_pol_rhs {
    template <typename T0, typename T1, typename T2>
    Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> operator()(const T0& t_in, const T1& y, std::ostream* msgs,
                                                                 const T2& theta) const {
      Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> res(2);
      auto& eps = theta[0];
      res << y(1),
        ((1 - y(0) * y(0)) * y(1) - y(0))/eps;
      return res;
    }
  };

  using T_init = std::tuple_element_t<2, T>;
  using T_param = std::tuple_element_t<3, T>;

  stan::math::nested_rev_autodiff nested;
  van_der_pol_rhs f;
  std::tuple_element_t<0, T> s1;
  std::tuple_element_t<1, T> s2;
  Eigen::Matrix<T_init, -1, 1> y0;
  std::vector<T_param> theta;
  double t0;
  std::vector<double> ts;
  double rtol;
  double atol;
  int max_num_step;

  van_der_pol_base() : nested(),
                                f(), s1(), s2(),
                                y0(2),
                                theta{1.e-4},
                                t0(0.0),
                                ts{11},
                                rtol(1.e-6),
                                atol(1.e-6),
                                max_num_step(100000) {
    y0(0) = 2.0;
    y0(1) = 0.0;
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
struct van_der_pol_benchmark_tol_1_test : public van_der_pol_base<T>,
                                             public ODEBenchmarkFixture<van_der_pol_benchmark_tol_1_test<T>> {
  van_der_pol_benchmark_tol_1_test() {
    this -> rtol = 1.e-3;
    this -> atol = 1.e-3;
  }
};

template <typename T>
struct van_der_pol_benchmark_tol_2_test : public van_der_pol_base<T>,
                                             public ODEBenchmarkFixture<van_der_pol_benchmark_tol_2_test<T>> {
  van_der_pol_benchmark_tol_2_test() {
    this -> rtol = 1.e-6;
    this -> atol = 1.e-6;
  }
};

template <typename T>
struct van_der_pol_benchmark_tol_3_test : public van_der_pol_base<T>,
                                             public ODEBenchmarkFixture<van_der_pol_benchmark_tol_3_test<T>> {
  van_der_pol_benchmark_tol_3_test() {
    this -> rtol = 1.e-8;
    this -> atol = 1.e-8;
  }
};

template <typename T>
struct van_der_pol_benchmark_tol_4_test : public van_der_pol_base<T>,
                                             public ODEBenchmarkFixture<van_der_pol_benchmark_tol_4_test<T>> {
  van_der_pol_benchmark_tol_4_test() {
    this -> rtol = 1.e-12;
    this -> atol = 1.e-12;
  }
};

#endif
