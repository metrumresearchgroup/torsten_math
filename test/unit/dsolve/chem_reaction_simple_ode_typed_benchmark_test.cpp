#include <stan/math/rev.hpp>
#include <boost/mp11.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/functor/test_fixture_ode.hpp>
#include <stan/math/torsten/test/unit/dsolve/test_fixture_chem_reaction_simple.hpp>
#include <stan/math/torsten/test/unit/dsolve/ode_test_functors.hpp>

using tolerance_1_test_types = boost::mp11::mp_product<
  std::tuple,
  ::testing::Types<pmx_ode_dirk3_functor, pmx_ode_dirk2_functor, pmx_ode_bdf_functor, pmx_ode_rk45_functor>,
  ::testing::Types<pmx_ode_bdf_functor>,
  ::testing::Types<stan::math::var_value<double>>, // init
  ::testing::Types<stan::math::var_value<double>>  // theta
  >;

TYPED_TEST_SUITE_P(chem_reaction_simple_benchmark_tolerance_1_test);
TYPED_TEST_P(chem_reaction_simple_benchmark_tolerance_1_test, solver_speed) {
  for (auto i = 0 ; i < 100; ++i) {
    auto sol = this -> apply_solver_ctrl(this -> solver1());
  }
}

REGISTER_TYPED_TEST_SUITE_P(chem_reaction_simple_benchmark_tolerance_1_test, solver_speed);
INSTANTIATE_TYPED_TEST_SUITE_P(StanOde, chem_reaction_simple_benchmark_tolerance_1_test, tolerance_1_test_types);


using tolerance_2_test_types = boost::mp11::mp_product<
  std::tuple,
  ::testing::Types<pmx_ode_dirk5_functor, pmx_ode_dirk5_functor, pmx_ode_bdf_functor>,
  ::testing::Types<pmx_ode_bdf_functor>,
  ::testing::Types<stan::math::var_value<double>>, // init
  ::testing::Types<stan::math::var_value<double>>  // theta
  >;

TYPED_TEST_SUITE_P(chem_reaction_simple_benchmark_tolerance_2_test);
TYPED_TEST_P(chem_reaction_simple_benchmark_tolerance_2_test, solver_speed) {
  for (auto i = 0 ; i < 100; ++i) {
    auto sol = this -> apply_solver_ctrl(this -> solver1());
  }
}

REGISTER_TYPED_TEST_SUITE_P(chem_reaction_simple_benchmark_tolerance_2_test, solver_speed);
INSTANTIATE_TYPED_TEST_SUITE_P(StanOde, chem_reaction_simple_benchmark_tolerance_2_test, tolerance_2_test_types);


using tolerance_3_test_types = boost::mp11::mp_product<
  std::tuple,
  ::testing::Types<pmx_ode_dirk5_functor, pmx_ode_dirk4_functor, pmx_ode_bdf_functor, pmx_ode_adams_functor>,
  ::testing::Types<pmx_ode_bdf_functor>,
  ::testing::Types<stan::math::var_value<double>>, // init
  ::testing::Types<stan::math::var_value<double>>  // theta
  >;

TYPED_TEST_SUITE_P(chem_reaction_simple_benchmark_tolerance_3_test);
TYPED_TEST_P(chem_reaction_simple_benchmark_tolerance_3_test, solver_speed) {
  for (auto i = 0 ; i < 100; ++i) {
    auto sol = this -> apply_solver_ctrl(this -> solver1());
  }
}

REGISTER_TYPED_TEST_SUITE_P(chem_reaction_simple_benchmark_tolerance_3_test, solver_speed);
INSTANTIATE_TYPED_TEST_SUITE_P(StanOde, chem_reaction_simple_benchmark_tolerance_3_test, tolerance_3_test_types);

