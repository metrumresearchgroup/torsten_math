#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/torsten/test/unit/pmx_ode_test_fixture.hpp>
#include <stan/math/torsten/test/unit/pmx_onecpt_test_fixture.hpp>
#include <stan/math/torsten/test/unit/pmx_twocpt_test_fixture.hpp>
#include <stan/math/torsten/test/unit/pmx_friberg_karlsson_test_fixture.hpp>
#include <stan/math/torsten/test/unit/expect_near_matrix_eq.hpp>
#include <stan/math/torsten/test/unit/expect_matrix_eq.hpp>
#include <stan/math/torsten/test/unit/pmx_twocpt_functor_with_data.hpp>
#include <stan/math/torsten/to_var.hpp>
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

template <typename EM, std::size_t... Is>
auto apply_to_tuple(const EM& em, std::integer_sequence<std::size_t, Is...> ) {
  return std::tie(em.template get_model_array_1d_param<Is>(0)...);
}

TEST_F(TorstenOneCptTest, variadic_real_data) {
  using stan::math::var;

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

  std::vector<std::vector<var> > pMatrix_var(torsten::to_var(pMatrix));  

  std::vector<std::vector<double> > x_r(time.size(), {1.0, 2.0});
  std::vector<std::vector<int> > x_i(time.size(), {1, 2, 3});
  using ER = torsten::NONMENEventsRecord<double, double, double, double>;
  using EM_1 = torsten::EventsManager<ER, NonEventParameters<double, var, std::vector,
                                                             std::tuple<double, double>, double>>;
  using EM_2 = torsten::EventsManager<ER, NonEventParameters<double, var, std::vector,
                                                             std::tuple<double, double>, double, int>>;
  
  const ER events_rec(nCmt, time, amt, rate, ii, evid, cmt, addl, ss);

  const EM_1 em1(0, events_rec, pMatrix_var, biovar, tlag, x_r);
  using x_r_1d_ref = typename std::tuple_element<0, decltype(apply_to_tuple(em1, std::make_integer_sequence<std::size_t, 1>{}))>::type;
  static_assert(std::is_same<x_r_1d_ref, const std::vector<double>&>::value, "");
  EXPECT_FLOAT_EQ(std::get<0>(apply_to_tuple(em1, std::make_integer_sequence<std::size_t, 1>{}))[0],
                  1.0);
  

  const EM_2 em2(0, events_rec, pMatrix_var, biovar, tlag, x_r, x_i);
  using x_i_1d_ref = typename std::tuple_element<1, decltype(apply_to_tuple(em2, std::make_integer_sequence<std::size_t, 2>{}))>::type;
  static_assert(std::is_same<x_i_1d_ref, const std::vector<int>&>::value, "");
  EXPECT_EQ(std::get<1>(apply_to_tuple(em2, std::make_integer_sequence<std::size_t, 2>{}))[2],
            3);
}

// TEST_F(TorstenOneCptTest, variadic_real_data) {
//   using stan::math::var;

//   resize(3);
//   time[0] = 0.0;
//   time[1] = 0.0;
//   for(int i = 2; i < nt; i++) time[i] = time[i - 1] + 5;

//   amt[0] = 1200;
//   addl[0] = 2;
//   ss[0] = 1;

//   double rtol = 1e-12, atol = 1e-12;
//   long int max_num_steps = 1e8;

//   biovar[0] = std::vector<double>{1.0, 1.0};
//   tlag[0] = std::vector<double>{0.0, 0.0};

//   std::vector<std::vector<var> > pMatrix_var(torsten::to_var(pMatrix));  

//   double cl_add = 1.2;
//   std::vector<std::vector<var> > theta_1(torsten::to_var(pMatrix));
//   theta_1[0][0] += cl_add;
//   std::vector<std::vector<double> > x_r{{cl_add}};
//   // auto y1 = pmx_solve_bdf(f_twocpt, nCmt, time, amt, rate,
//   //                         ii, evid, cmt, addl, ss, theta_1, biovar, tlag,
//   //                         rtol, atol, max_num_steps, rtol, atol, max_num_steps);
//   // auto y2 = pmx_solve_bdf(f_twocpt, nCmt, time, amt, rate,
//   //                         ii, evid, cmt, addl, ss, pMatrix_var, biovar, tlag, x_r,
//   //                         rtol, atol, max_num_steps, rtol, atol, max_num_steps);
//   // torsten::test::test_grad(theta_1[0], pMatrix_var[0], y1, y2, 1e-5, 1e-6);

//   // auto f3 = [&] (const std::vector<double>& x) {
//   //             return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
//   //                                   ii, evid, cmt, addl, ss, x, biovar[0], tlag[0], rtol, atol,
//   //             max_num_steps);
//   //           };
//   // auto f4 = [&] (const std::vector<stan::math::var>& x) {
//   //             return pmx_solve_rk45(f_onecpt, nCmt, time, amt, rate,
//   //                                   ii, evid, cmt, addl, ss, x, biovar, rtol, atol,
//   //             max_num_steps);
//   //           };
//   // torsten::test::test_grad(f3, f4, pMatrix[0], 2e-5, 1e-6, 1e-5, 1e-6);
// }
