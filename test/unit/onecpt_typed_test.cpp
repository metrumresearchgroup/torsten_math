#include <stan/math/torsten/test/unit/test_fixture_onecpt.hpp>
#include <stan/math/torsten/test/unit/test_functors.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <boost/mp11.hpp>

TYPED_TEST_SUITE_P(test_onecpt);
TYPED_TEST_P(test_onecpt, multiple_bolus) {
  Eigen::MatrixXd amounts(10, 2);
  amounts << 1000.0, 0.0,
    740.8182, 254.97490,
    548.8116, 436.02020,
    406.5697, 562.53846,
    301.1942, 648.89603,
    223.1302, 705.72856,
    165.2989, 740.90816,
    122.4564, 760.25988,
    90.71795, 768.09246,
    8.229747, 667.87079;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x);

  this -> biovar[0] = {0.8, 0.9};
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-8, "RATE");
  this -> compare_solvers_adj(this -> ii, 1.e-8, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> tlag[0], 1.e-8, "lag time");
}

TYPED_TEST_P(test_onecpt, multiple_bolus_addl) {
  this -> ii[0] = 1.3;          // ensure test II + ADDL by end of time
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-8, "RATE");
  this -> compare_solvers_adj(this -> ii, 5.e-6, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> tlag[0], 1.e-8, "lag time");
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

  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-8, "RATE");
  this -> compare_solvers_adj(this -> ii, 5.e-6, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> tlag[0], 5.e-7, "lag time");
}

TYPED_TEST_P(test_onecpt, multiple_infusion) {
  this -> reset_events(3);
  this -> amt[0] = 1200;
  this -> rate[0] = 100;
  this -> addl[0] = 0;
  this -> ii[0] = 17.0;
  this -> ss[0] = 1;
  this -> time[0] = 0.0;
  this -> time[1] = 17.0 * 0.5;
  this -> time[2] = 17.0;

  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-8, "RATE");
  this -> compare_solvers_adj(this -> ii, 5.e-6, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> tlag[0], 1.e-8, "lag time");  
}

REGISTER_TYPED_TEST_SUITE_P(test_onecpt,
                            multiple_bolus,
                            multiple_bolus_addl,
                            single_bolus_with_tlag,
                            multiple_infusion);

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
