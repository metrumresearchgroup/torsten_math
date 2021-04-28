#include <stan/math/torsten/test/unit/test_fixture_onecpt.hpp>
#include <stan/math/torsten/test/unit/test_functors.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <boost/mp11.hpp>

TYPED_TEST_SUITE_P(test_onecpt);
TYPED_TEST_P(test_onecpt, multiple_bolus) {
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-8, "RATE");
  this -> compare_solvers_adj(this -> ii, 1.e-8, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> tlag[0], 1.e-8, "lag time");
}

REGISTER_TYPED_TEST_SUITE_P(test_onecpt, multiple_bolus);

using onecpt_test_types = boost::mp11::mp_product<
  std::tuple,
  ::testing::Types<pmx_solve_onecpt_functor>, // solver 1
  ::testing::Types<pmx_solve_linode_functor,
                   pmx_solve_rk45_functor,
                   pmx_solve_bdf_functor,
                   pmx_solve_adams_functor>, // solver 2
  ::testing::Types<double, stan::math::var_value<double>>,  // TIME
  ::testing::Types<double, stan::math::var_value<double>>,  // AMT
  ::testing::Types<double, stan::math::var_value<double>> , // RATE
  ::testing::Types<double, stan::math::var_value<double>> , // II
  ::testing::Types<double, stan::math::var_value<double>> , // PARAM
  ::testing::Types<double, stan::math::var_value<double>> , // BIOVAR
  ::testing::Types<double, stan::math::var_value<double>> , // TLAG
  ::testing::Types<torsten::PMXOneCptODE>                   // ODE
    >;

INSTANTIATE_TYPED_TEST_SUITE_P(PMX, test_onecpt, onecpt_test_types);
