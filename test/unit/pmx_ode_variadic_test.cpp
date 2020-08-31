#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/torsten/test/unit/pmx_ode_test_fixture.hpp>
#include <stan/math/torsten/test/unit/pmx_onecpt_test_fixture.hpp>
#include <stan/math/torsten/test/unit/pmx_twocpt_test_fixture.hpp>
#include <stan/math/torsten/test/unit/pmx_friberg_karlsson_test_fixture.hpp>
#include <stan/math/torsten/test/unit/expect_near_matrix_eq.hpp>
#include <stan/math/torsten/test/unit/expect_matrix_eq.hpp>
#include <stan/math/torsten/pmx_solve_rk45.hpp>
#include <stan/math/torsten/pmx_solve_bdf.hpp>
#include <stan/math/torsten/pmx_solve_adams.hpp>
#include <stan/math/torsten/pmx_twocpt_model.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <stan/math/torsten/test/unit/util_generalOdeModel.hpp>
#include <gtest/gtest.h>

auto f_onecpt = torsten::PMXOneCptModel<double>::f_;
auto f_twocpt = torsten::PMXTwoCptModel<double>::f_;

using stan::math::var;
using std::vector;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Dynamic;

using torsten::pmx_solve_rk45;
using torsten::pmx_solve_bdf;
using torsten::pmx_solve_adams;
using torsten::NONMENEventsRecord;
using torsten::NonEventParameters;

TEST_F(TorstenOneCptTest, variadic_tlag) {
  resize(3);
  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < nt; i++) time[i] = time[i - 1] + 5;

  amt[0] = 1200;
  addl[0] = 2;
  ss[0] = 1;

  double rtol = 1e-12, atol = 1e-12;
  long int max_num_steps = 1e8;

  biovar[0] = std::vector<double>{1.0, 1.0};
  tlag[0] = std::vector<double>{0.0, 0.0};

  auto f1 = [&] (const std::vector<std::vector<double> >& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
              ii, evid, cmt, addl, ss, x, biovar, tlag, rtol, atol,
              max_num_steps);
            };
  auto f2 = [&] (const std::vector<std::vector<stan::math::var> >& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
              ii, evid, cmt, addl, ss, x, biovar, rtol, atol,
              max_num_steps);
            };
  torsten::test::test_grad(f1, f2, pMatrix, 2e-5, 1e-6, 1e-5, 1e-6);

  auto f3 = [&] (const std::vector<double>& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
                                    ii, evid, cmt, addl, ss, x, biovar[0], tlag[0], rtol, atol,
              max_num_steps);
            };
  auto f4 = [&] (const std::vector<stan::math::var>& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
                                    ii, evid, cmt, addl, ss, x, biovar, rtol, atol,
              max_num_steps);
            };
  torsten::test::test_grad(f3, f4, pMatrix[0], 2e-5, 1e-6, 1e-5, 1e-6);
}

TEST_F(TorstenOneCptTest, variadic_biovar) {
  resize(3);
  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < nt; i++) time[i] = time[i - 1] + 5;

  amt[0] = 1200;
  addl[0] = 2;
  ss[0] = 1;

  double rtol = 1e-12, atol = 1e-12;
  long int max_num_steps = 1e8;

  biovar[0] = std::vector<double>{1.0, 1.0};
  tlag[0] = std::vector<double>{0.0, 0.0};

  auto f1 = [&] (const std::vector<std::vector<double> >& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
              ii, evid, cmt, addl, ss, x, biovar, tlag, rtol, atol,
              max_num_steps);
            };
  auto f2 = [&] (const std::vector<std::vector<stan::math::var> >& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
              ii, evid, cmt, addl, ss, x, rtol, atol,
              max_num_steps);
            };
  torsten::test::test_grad(f1, f2, pMatrix, 2e-5, 1e-6, 1e-5, 1e-6);

  auto f3 = [&] (const std::vector<double>& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
                                    ii, evid, cmt, addl, ss, x, biovar[0], tlag[0], rtol, atol,
              max_num_steps);
            };
  auto f4 = [&] (const std::vector<stan::math::var>& x) {
              return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
                                    ii, evid, cmt, addl, ss, x, rtol, atol,
              max_num_steps);
            };
  torsten::test::test_grad(f3, f4, pMatrix[0], 2e-5, 1e-6, 1e-5, 1e-6);
}
