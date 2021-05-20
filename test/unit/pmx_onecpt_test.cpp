#include <stan/math/torsten/test/unit/test_macros.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/test/unit/pmx_onecpt_test_fixture.hpp>
#include <stan/math/torsten/pmx_solve_onecpt.hpp>
#include <stan/math/torsten/pmx_solve_bdf.hpp>
#include <stan/math/torsten/pmx_solve_rk45.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_twocpt_model.hpp>
#include <gtest/gtest.h>
#include <vector>

using std::vector;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Dynamic;
using torsten::pmx_solve_onecpt;
using torsten::pmx_solve_bdf;
using torsten::pmx_solve_rk45;

TEST_F(TorstenOneCptTest, multiple_bolus_overload) {
  resize(4);
  time[0] = 0;
  for(int i = 1; i < nt; i++) time[i] = time[i - 1] + 0.9;
  addl[0] = 1;

  biovar[0] = std::vector<double>{0.8, 0.9};
  tlag[0] = std::vector<double>{0.5, 0.8};
  TORSTEN_CPT_PARAM_OVERLOAD_TEST(pmx_solve_onecpt, time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag, 1e-6, 1e-6);
}

TEST_F(TorstenOneCptTest, steady_state_overload) {
  resize(3);
  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < nt; i++) time[i] = time[i - 1] + 5;
  amt[0] = 1200;
  addl[0] = 10;
  ss[0] = 1;

  tlag[0][0] = 1.7;  // tlag1
  tlag[0][1] = 0;  // tlag2
  TORSTEN_CPT_PARAM_OVERLOAD_TEST(pmx_solve_onecpt, time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag, 1e-6, 1e-6);
}

TEST_F(TorstenOneCptTest, multiple_steady_state_iv_overload) {
  resize(3);
  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < nt; i++) time[i] = time[i - 1] + 5;
  amt[0] = 1200;
  rate[0] = 150;
  addl[0] = 10;
  ss[0] = 1;
  std::vector<std::vector<double> > biovar_test(1, {0.8, 0.9});
  std::vector<std::vector<double> > tlag_test(1, {2.4, 1.7});
  TORSTEN_CPT_PARAM_OVERLOAD_TEST(pmx_solve_onecpt, time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag, 1e-6, 1e-6);
}

TEST_F(TorstenOneCptTest, single_iv_var_overload) {
  resize(2);
  amt[0] = 1200;
  rate[0] = 340;
  std::vector<stan::math::var> rate_v(stan::math::to_var(rate));
  TORSTEN_CPT_PARAM_OVERLOAD_TEST(pmx_solve_onecpt, time, amt, rate_v, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag, 1e-6, 1e-6);
}

TEST_F(TorstenOneCptTest, single_iv_central_cmt_var_optoinal_biovar_tlag) {
  cmt[0] = 2;  // IV infusion, not absorption from the gut
  rate[0] = 600;

  std::vector<stan::math::var> rate_var = stan::math::to_var(rate);
  auto x1 = pmx_solve_onecpt(time, amt, rate_var, ii, evid, cmt, addl, ss,
                                 pMatrix, biovar, tlag);
  auto x2 = pmx_solve_onecpt(time, amt, rate_var, ii, evid, cmt, addl, ss,
                                 pMatrix, biovar);
  auto x3 = pmx_solve_onecpt(time, amt, rate_var, ii, evid, cmt, addl, ss,
                                 pMatrix);

  torsten::test::test_grad(rate_var, x1, x2, 1.e-12, 1.e-12);
  torsten::test::test_grad(rate_var, x1, x3, 1.e-12, 1.e-12);
}

TEST_F(TorstenOneCptTest, single_iv_central_cmt_var_overload) {
  resize(3);
  cmt[0] = 2;  // IV infusion, not absorption from the gut
  rate[0] = 660;
  std::vector<stan::math::var> rate_v(stan::math::to_var(rate));
  TORSTEN_CPT_PARAM_OVERLOAD_TEST(pmx_solve_onecpt, time, amt, rate_v, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag, 1e-6, 1e-6);
}

TEST_F(TorstenOneCptTest, reset_an_cmt) {
  nt = 3;
  resize(nt);
  evid[0] = 1;
  evid[1] = 2;
  evid[2] = 1;
  cmt[0] = 1;
  cmt[1] = -2;
  amt[2]= 800;
  ii[0] = 0;
  addl[0] = 0;
  time[0] = 0.0;
  time[2] = time[1];
  
  auto y = pmx_solve_onecpt(time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
  EXPECT_FLOAT_EQ(y(0, 1), 740.8182206817178);
  EXPECT_FLOAT_EQ(y(1, 1), 0.0);
  EXPECT_FLOAT_EQ(y(0, 2), 740.8182206817178);
  EXPECT_FLOAT_EQ(y(1, 2), 800.0);
}

/*
TEST(Torsten, pmx_solve_onecptModel_SS_rate_2) {
  // Test the special case where the infusion rate is longer than
  // the interdose interval.
  // THIS TEST FAILS.
  using std::vector;

  vector<vector<double> > pMatrix(1);
  pMatrix[0].resize(3);
  pMatrix[0][0] = 10;  // CL
  pMatrix[0][1] = 80;  // Vc
  pMatrix[0][2] = 1.2;  // ka

  int nCmt = 2;
  vector<vector<double> > biovar(1);
  biovar[0].resize(nCmt);
  biovar[0][0] = 1;  // F1
  biovar[0][1] = 1;  // F2

  vector<vector<double> > tlag(1);
  tlag[0].resize(nCmt);
  tlag[0][0] = 0;  // tlag1
  tlag[0][1] = 0;  // tlag2

  vector<double> time(10);
  time[0] = 0;
  for(int i = 1; i < 9; i++) time[i] = time[i - 1] + 0.25;
  time[9] = 4.0;

  vector<double> amt(10, 0);
  amt[0] = 1200;

  vector<double> rate(10, 0);
  rate[0] = 75;

  vector<int> cmt(10, 2);
  cmt[0] = 1;

  vector<int> evid(10, 0);
  evid[0] = 1;

  vector<double> ii(10, 0);
  ii[0] = 12;

  vector<int> addl(10, 0);
  addl[0] = 14;

  vector<int> ss(10, 0);

  MatrixXd x;
  x = pmx_solve_onecpt(time, amt, rate, ii, evid, cmt, addl, ss,
                    pMatrix, biovar, tlag);

  std::cout << x << std::endl;

  MatrixXd amounts(10, 2);
  amounts << 62.50420, 724.7889,
             78.70197, 723.4747,
             90.70158, 726.3310,
             99.59110, 732.1591,
             106.17663, 740.0744,
             111.05530, 749.4253,
             114.66951, 759.7325,
             117.34699, 770.6441,
             119.33051, 781.9027,
             124.48568, 870.0308;

  // expect_matrix_eq(amounts, x);

  // Test AutoDiff against FiniteDiff
  // test_pmx_solve_onecpt(time, amt, rate, ii, evid, cmt, addl, ss,
  //                   pMatrix, biovar, tlag, 1e-8, 5e-4);
} */
