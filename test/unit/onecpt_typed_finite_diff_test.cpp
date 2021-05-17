#include <stan/math/torsten/test/unit/test_fixture_onecpt.hpp>
#include <stan/math/torsten/test/unit/test_functors.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <boost/mp11.hpp>

TYPED_TEST_SUITE_P(test_onecpt);
TYPED_TEST_P(test_onecpt, multiple_bolus) {
  this -> test_finite_diff_amt(1.e-3, 1.e-6);
  this -> test_finite_diff_theta(1.e-5, 1.e-4);
}

TYPED_TEST_P(test_onecpt, single_bolus_with_tlag) {
  this -> reset_events(2);      // only need two events
  this -> amt[0] = 1000;
  this -> evid[0] = 1;
  this -> cmt[0] = 1;
  this -> ii[0] = 0;
  this -> addl[0] = 0;
  this -> time[0] = 0.0;
  this -> tlag[0][0] = 1.5;
  this -> time[1] = 2.5;

  this -> test_finite_diff_amt(1.e-3, 1.e-6);
  this -> test_finite_diff_theta(1.e-5, 1.e-4);
  this -> test_finite_diff_tlag(1.e-3, 1.e-4);
}

TYPED_TEST_P(test_onecpt, multiple_infusion) {
  this -> cmt[0] = 2;
  this -> rate[0] = 350;
  this -> addl[0] = 2;

  this -> biovar[0] = {0.8, 0.9};
  this -> tlag[0] = {0.4, 0.8};

  this -> test_finite_diff_amt(1.e-3, 1.e-6);
  this -> test_finite_diff_rate(1.e-3, 1.e-6);
  this -> test_finite_diff_theta(1.e-3, 1.e-6);
  this -> test_finite_diff_tlag(1.e-3, 1.e-6);
}

REGISTER_TYPED_TEST_SUITE_P(test_onecpt,
                            multiple_bolus,
                            single_bolus_with_tlag,
                            multiple_infusion);

using onecpt_test_types = boost::mp11::mp_product<
  std::tuple,
  ::testing::Types<pmx_solve_onecpt_functor>, // solver 1
  ::testing::Types<pmx_solve_onecpt_functor>, // solver 2
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
