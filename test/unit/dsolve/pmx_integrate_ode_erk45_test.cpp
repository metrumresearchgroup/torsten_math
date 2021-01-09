#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_erk45.hpp>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_group_erk45.hpp>
#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/torsten/mpi.hpp>
#include <test/unit/math/rev/fun/util.hpp>
#include <stan/math/torsten/test/unit/pmx_ode_test_fixture.hpp>
#include <test/unit/math/prim/functor/harmonic_oscillator.hpp>
#include <stan/math/rev/functor/integrate_ode_bdf.hpp>
#include <arkode/arkode_erkstep.h>
#include <nvector/nvector_serial.h>
#include <test/unit/util.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

using stan::math::integrate_ode_rk45;
using torsten::pmx_integrate_ode_erk45;
using torsten::dsolve::PMXArkodeSystem;
using torsten::dsolve::Arkode;
using torsten::dsolve::PMXOdeService;
using torsten::dsolve::PMXArkodeIntegrator;
using torsten::pmx_integrate_ode_group_rk45;
using stan::math::var;
using std::vector;

#if defined(STAN_LANG_MPI) || defined(TORSTEN_MPI)
TORSTEN_MPI_SESSION_INIT;
#endif

TEST_F(TorstenOdeTest_sho, arkode_erk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_erk45(f, y0, t0, ts, theta, x_r, x_i));
  for (size_t i = 0; i < y1.size(); ++i) {
    for (size_t j = 0; j < y1[i].size(); ++j) {
      EXPECT_NEAR(y1[i][j], y2[i][j], 5.e-6);
    }
  }
}

TEST_F(TorstenOdeTest_sho, arkode_erk45_ivp_system_matrix_result) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta, x_r, x_i, msgs, atol, rtol, max_num_steps));

  using Ode = PMXArkodeSystem<harm_osc_ode_fun, double, double, double>;
  PMXOdeService<Ode> serv(y0.size(), theta.size());
  Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
  PMXArkodeIntegrator<DORMAND_PRINCE_7_4_5> solver(rtol, atol, max_num_steps);

  Eigen::MatrixXd y2 = solver.integrate<Ode, false>(ode);

  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_lorenz, arkode_erk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_erk45(f, y0, t0, ts, theta , x_r, x_i));
  for (size_t i = 0; i < y1.size(); ++i) {
    for (size_t j = 0; j < y1[i].size(); ++j) {
      EXPECT_NEAR(y1[i][j], y2[i][j], 1.5e-4);
    }
  }
}

TEST_F(TorstenOdeTest_sho, arkode_erk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  ts.resize(1);
  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.5E-8);
}

TEST_F(TorstenOdeTest_lorenz, arkode_erk45_ivp_system_matrix_result) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta, x_r, x_i, msgs, atol, rtol, max_num_steps));

  using Ode = PMXArkodeSystem<lorenz_ode_fun, double, double, double>;
  PMXOdeService<Ode> serv(y0.size(), theta.size());
  Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
  PMXArkodeIntegrator<DORMAND_PRINCE_7_4_5> solver(rtol, atol, max_num_steps);

  Eigen::MatrixXd y2 = solver.integrate<Ode, false>(ode);

  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_chem, arkode_erk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_erk45(f, y0, t0, ts, theta , x_r, x_i));
  for (size_t i = 0; i < y1.size(); ++i) {
    for (size_t j = 0; j < y1[i].size(); ++j) {
      EXPECT_NEAR(y1[i][j], y2[i][j], 1.6e-4);
    }
  }
}

TEST_F(TorstenOdeTest_chem, arkode_erk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 3.E-10, 7.E-6);
}

TEST_F(TorstenOdeTest_sho, fwd_sensitivity_ts) {
  using stan::math::value_of;
  std::vector<var> ts_var = stan::math::to_var(ts);

  std::vector<std::vector<var> > y, y1, y2;
  y = pmx_integrate_ode_erk45(f, y0, t0, ts_var, theta, x_r, x_i, msgs);

  std::vector<double> g(ts.size()), fval(y0.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    fval = f(value_of(ts[i]), value_of(y[i]), theta, x_r, x_i, msgs);
    for (size_t j = 0; j < y0.size(); ++j) {
      stan::math::set_zero_all_adjoints();
      y[i][j].grad(ts_var, g);
      for (size_t k = 0; k < ts.size(); ++k) {
        if (k == i) {
          EXPECT_FLOAT_EQ(g[k], fval[j]);
        } else {
          EXPECT_FLOAT_EQ(g[k], 0.0);
        }
      }
    }
  }
}

TEST_F(TorstenOdeTest_sho, rk45_theta_var_matrix_result) {
  std::vector<var> theta_var = stan::math::to_var(theta);
  vector<vector<var> > y1(integrate_ode_rk45(f, y0, t0, ts, theta_var, x_r, x_i, msgs, atol, rtol, max_num_steps));

  using Ode = PMXArkodeSystem<harm_osc_ode_fun, double, double, var>;
  PMXOdeService<Ode> serv(y0.size(), theta.size());
  Ode ode{serv, f, t0, ts, y0, theta_var, x_r, x_i, msgs};
  PMXArkodeIntegrator<DORMAND_PRINCE_7_4_5> solver(rtol, atol, max_num_steps);
  Eigen::MatrixXd y2 = solver.integrate<Ode, false>(ode);

  torsten::test::test_grad(theta_var, y1, y2, 1.e-8, 1.e-8);
}

TEST_F(TorstenOdeTest_lorenz, arkode_erk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 3.E-5, 1.3E-4);
}

TEST_F(TorstenOdeTest_chem, arkode_erk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 5.0E-11, 1.E-6);
}

TEST_F(TorstenOdeTest_lorenz, arkode_erk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-4, 1.E-4);
}

TEST_F(TorstenOdeTest_sho, arkode_erk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 3.E-6, 2.5E-6);
}

TEST_F(TorstenOdeTest_sho, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-6, 1.E-6);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-6, 5.E-6);
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.5E-5, 6.E-5);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.5E-5, 2.5E-4);
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_erk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 2.E-7);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 2.E-6);
}

TEST_F(TorstenOdeTest_neutropenia, sequential_group_erk45_fwd_sensitivity_theta) {
  // size of population
  const int np = 2;

  vector<var> theta_var = stan::math::to_var(theta);

  vector<int> len(np, ts.size());
  vector<double> ts_m;
  ts_m.reserve(np * ts.size());
  for (int i = 0; i < np; ++i) ts_m.insert(ts_m.end(), ts.begin(), ts.end());

  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, stan::math::to_var(theta));
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = pmx_integrate_ode_erk45(f, y0, t0, ts, theta_var, x_r, x_i);
  Eigen::Matrix<var, -1, -1> y_m1 = pmx_integrate_ode_group_erk45(f, y0_m, t0, len, ts_m, theta_var_m , x_r_m, x_i_m);
  Eigen::Matrix<var, -1, -1> y_m2 = pmx_integrate_ode_group_erk45(f, y0_m, t0, len, ts_m, theta_var_m , x_r_m, x_i_m);

  EXPECT_EQ(y_m1.cols(), ts_m.size());
  EXPECT_EQ(y_m2.cols(), ts_m.size());
  int icol = 0;
  for (int i = 0; i < np; ++i) {
    stan::math::matrix_v y_i = y_m1.block(0, icol, y0.size(), len[i]);
    torsten::test::test_grad(theta_var, theta_var_m[i], y, y_i, 1e-16, 1e-10);
    icol += len[i];
  }

  icol = 0;
  for (int i = 0; i < np; ++i) {
    stan::math::matrix_v y_i = y_m2.block(0, icol, y0.size(), len[i]);
    torsten::test::test_grad(theta_var, theta_var_m[i], y, y_i, 1e-16, 1e-10);
    icol += len[i];
  }
}

TEST_F(TorstenOdeTest_neutropenia, sequential_group_rk45_fwd_sensitivity_theta) {
  // size of population
  const int np = 2;

  vector<var> theta_var = stan::math::to_var(theta);

  vector<int> len(np, ts.size());
  vector<double> ts_m;
  ts_m.reserve(np * ts.size());
  for (int i = 0; i < np; ++i) ts_m.insert(ts_m.end(), ts.begin(), ts.end());

  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, stan::math::to_var(theta));
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = integrate_ode_rk45(f, y0, t0, ts, theta_var, x_r, x_i);
  Eigen::Matrix<var, -1, -1> y_m1 = pmx_integrate_ode_group_erk45(f, y0_m, t0, len, ts_m, theta_var_m , x_r_m, x_i_m);
  Eigen::Matrix<var, -1, -1> y_m2 = pmx_integrate_ode_group_erk45(f, y0_m, t0, len, ts_m, theta_var_m , x_r_m, x_i_m);

  EXPECT_EQ(y_m1.cols(), ts_m.size());
  EXPECT_EQ(y_m2.cols(), ts_m.size());
  int icol = 0;
  for (int i = 0; i < np; ++i) {
    stan::math::matrix_v y_i = y_m1.block(0, icol, y0.size(), len[i]);
    torsten::test::test_grad(theta_var, theta_var_m[i], y, y_i, 5e-6, 1.5e-5);
    icol += len[i];
  }

  icol = 0;
  for (int i = 0; i < np; ++i) {
    stan::math::matrix_v y_i = y_m2.block(0, icol, y0.size(), len[i]);
    torsten::test::test_grad(theta_var, theta_var_m[i], y, y_i, 5e-6, 1.5e-5);
    icol += len[i];
  }
}
