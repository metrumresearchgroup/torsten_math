#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/dsolve/pmx_ode_dirk5.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_ckrk.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_rk45.hpp>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_group_rk45.hpp>
#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/torsten/mpi.hpp>
#include <stan/math/torsten/test/unit/test_util.hpp>
#include <test/unit/math/rev/fun/util.hpp>
#include <stan/math/torsten/test/unit/pmx_ode_test_fixture.hpp>
#include <test/unit/math/prim/functor/harmonic_oscillator.hpp>
#include <stan/math/rev/functor/integrate_ode_bdf.hpp>
#include <nvector/nvector_serial.h>
#include <test/unit/util.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

using stan::math::ode_ckrk;
using stan::math::ode_rk45;
using torsten::pmx_ode_dirk5;
using torsten::dsolve::PMXVariadicOdeSystem;
using torsten::dsolve::PMXOdeintIntegrator;
using torsten::pmx_integrate_ode_group_rk45;
using stan::math::var;
using std::vector;
using dsolve::OdeObserver;
using dsolve::OdeDataObserver;

#if defined(STAN_LANG_MPI) || defined(TORSTEN_MPI)
TORSTEN_MPI_SESSION_INIT;
#endif

// TEST_F(TorstenOdeTest_neutropenia, dirk5_adj_y0) {
//   stan::math::nested_rev_autodiff nested;  

//   std::vector<stan::math::var> theta_var = stan::math::to_var(theta);
//   Eigen::Matrix<stan::math::var, -1, 1>  y0_vec_var = stan::math::to_var(y0_vec);

//   auto y1 = pmx_ode_dirk5(f_eigen, y0_vec_var, t0, ts, nullptr, theta, x_r, x_i);    
//   auto y2 = ode_bdf(f_eigen, y0_vec_var, t0, ts, nullptr, theta, x_r, x_i);    

//   EXPECT_ARRAY2D_ADJ_NEAR(y1, y2, y0_vec_var, nested, 1.e-6, "");
// }

// TEST_F(TorstenOdeTest_neutropenia, dirk5_adj_theta) {
//   stan::math::nested_rev_autodiff nested;  

//   std::vector<stan::math::var> theta_var = stan::math::to_var(theta);
//   Eigen::Matrix<stan::math::var, -1, 1>  y0_vec_var = stan::math::to_var(y0_vec);

//   auto y1 = pmx_ode_dirk5(f_eigen, y0_vec, t0, ts, nullptr, theta_var, x_r, x_i);    
//   auto y2 = ode_bdf(f_eigen, y0_vec, t0, ts, nullptr, theta_var, x_r, x_i);    

//   EXPECT_ARRAY2D_ADJ_NEAR(y1, y2, theta_var, nested, 1.e-6, "");
// }

// TEST_F(TorstenOdeTest_neutropenia, dirk5_adj_y0_theta) {
//   stan::math::nested_rev_autodiff nested;  

//   std::vector<stan::math::var> theta_var = stan::math::to_var(theta);
//   Eigen::Matrix<stan::math::var, -1, 1>  y0_vec_var = stan::math::to_var(y0_vec);

//   auto y1 = pmx_ode_dirk5(f_eigen, y0_vec_var, t0, ts, nullptr, theta_var, x_r, x_i);    
//   auto y2 = ode_bdf(f_eigen, y0_vec_var, t0, ts, nullptr, theta_var, x_r, x_i);    

//   EXPECT_ARRAY2D_ADJ_NEAR(y1, y2, y0_vec_var, nested, 1.e-6, "");
//   EXPECT_ARRAY2D_ADJ_NEAR(y1, y2, theta_var, nested, 1.e-6, "");
// }

// TEST_F(TorstenOdeTest_chem, dirk5_adj_theta) {
//   stan::math::nested_rev_autodiff nested;  

//   std::vector<stan::math::var> theta_var = stan::math::to_var(theta);

//   auto y1 = pmx_ode_dirk5(f_eigen, y0_vec, t0, ts, nullptr, theta_var, x_r, x_i);    
//   auto y2 = ode_bdf(f_eigen, y0_vec, t0, ts, nullptr, theta_var, x_r, x_i);

//   EXPECT_ARRAY2D_ADJ_NEAR(y1, y2, theta_var, nested, 1.e-6, "theta");
// }

TEST_F(TorstenOdeTest_chem, dirk5_adj_theta_1) {
  stan::math::nested_rev_autodiff nested;  

  std::vector<stan::math::var> theta_var = stan::math::to_var(theta);
  Eigen::Matrix<stan::math::var, -1, 1>  y0_vec_var = stan::math::to_var(y0_vec);

  for (auto i = 0; i < 100; ++i) {
    auto y2 = pmx_ode_dirk5(f_eigen, y0_vec_var, t0, ts, nullptr, theta_var, x_r, x_i);    
  }
}

TEST_F(TorstenOdeTest_chem, dirk5_adj_theta_2) {
  stan::math::nested_rev_autodiff nested;  

  std::vector<stan::math::var> theta_var = stan::math::to_var(theta);
  Eigen::Matrix<stan::math::var, -1, 1>  y0_vec_var = stan::math::to_var(y0_vec);

  for (auto i = 0; i < 100; ++i) {
    auto y2 = ode_bdf_tol(f_eigen, y0_vec_var, t0, ts, 1.e-9, 1.e-9, 100000, nullptr, theta_var, x_r, x_i);    
  }
}
