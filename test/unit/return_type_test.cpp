#include <gtest/gtest.h>
#include <stan/math/prim/meta/is_var.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/torsten/meta/is_std_ode.hpp>
#include <stan/math/torsten/meta/is_eigen_ode.hpp>
#include <stan/math/torsten/meta/is_nl_system.hpp>
#include <test/unit/math/prim/functor/harmonic_oscillator.hpp>
#include <test/unit/math/rev/functor/util_algebra_solver.hpp>

TEST(Torsten, ode_signature) {
  EXPECT_TRUE((torsten::is_std_ode<harm_osc_ode_fun>::value));
  EXPECT_FALSE((torsten::is_std_ode<harm_osc_ode_fun_eigen>::value));
  EXPECT_TRUE((torsten::is_eigen_ode<harm_osc_ode_fun_eigen,
               std::vector<double>, std::vector<double>, std::vector<int>>::value));
  EXPECT_FALSE((torsten::is_eigen_ode<harm_osc_ode_fun,
                std::vector<double>, std::vector<double>, std::vector<int>>::value));
}

using torsten::nl_system_adaptor;
TEST(Torsten, nonlinear_system_signature) {
  EXPECT_TRUE((torsten::is_nl_system<nl_system_adaptor<simple_eq_functor>,
               Eigen::VectorXd, std::vector<double>, std::vector<int>>::value));
  EXPECT_TRUE((torsten::is_nl_system<nl_system_adaptor<non_linear_eq_functor>,
               Eigen::VectorXd, std::vector<double>, std::vector<int>>::value));
  EXPECT_FALSE((torsten::is_nl_system<simple_eq_functor,
               Eigen::VectorXd, std::vector<double>, std::vector<int>>::value));
  EXPECT_FALSE((torsten::is_nl_system<non_linear_eq_functor,
               Eigen::VectorXd, std::vector<double>, std::vector<int>>::value));
}
