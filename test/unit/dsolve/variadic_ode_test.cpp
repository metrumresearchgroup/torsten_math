#include <gtest/gtest.h>
#include <stan/math/prim/meta/is_var.hpp>
#include <stan/math/rev/meta/is_var.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_system.hpp>
#include <test/unit/math/prim/functor/harmonic_oscillator.hpp>

TEST(Torsten, variadic_ode_system) {
  using torsten::dsolve::PMXVariadicOdeSystem;
  using torsten::dsolve::PMXOdeSystem;
  using stan::math::value_of;
  Eigen::Matrix<stan::math::var, -1, 1> y0_var(2);
  y0_var << 1.0, 0.5;

  std::vector<stan::math::var> theta_var(1);
  theta_var[0] = 0.15;
  std::vector<double> ts{1.0, 2.0};

  std::vector<double> x_r;
  std::vector<int> x_i;

  harm_osc_ode_fun_eigen f0;
  PMXVariadicOdeSystem<harm_osc_ode_fun_eigen, double, stan::math::var,
                       std::vector<stan::math::var>, std::vector<double>, std::vector<int>> 
    ode(f0, 0.0, ts, y0_var, nullptr, theta_var, x_r, x_i);

  Eigen::VectorXd y0(value_of(y0_var));  
  Eigen::VectorXd dydt = f0(ts[0], y0, nullptr, value_of(theta_var), x_r, x_i);

  N_Vector nv_y(N_VNew_Serial(2));
  N_Vector ydot(N_VNew_Serial(2));
  NV_Ith_S(nv_y, 0) = y0[0];
  NV_Ith_S(nv_y, 1) = y0[1];
  ode(ts[0], nv_y, ydot);
  EXPECT_FLOAT_EQ(NV_Ith_S(ydot, 0), dydt[0]);
  EXPECT_FLOAT_EQ(NV_Ith_S(ydot, 1), dydt[1]);

  PMXOdeSystem<harm_osc_ode_fun, double, stan::math::var, stan::math::var>
    ode0(harm_osc_ode_fun(), 0.0, ts, stan::math::to_array_1d(y0_var),
         theta_var, x_r, x_i, nullptr);
  std::vector<double> dydt_vec(ode.system_size);
  ode0(ode0.y0_fwd_system, dydt_vec, ts[0]);

  dydt.resize(ode.system_size);
  ode(ode.y0_fwd_system, dydt, ts[0]);
  EXPECT_EQ(ode.system_size, ode0.system_size);
  for (size_t i = 0; i < ode0.system_size; ++i) {
    EXPECT_FLOAT_EQ(dydt(i), dydt_vec[i]);
  }
}
