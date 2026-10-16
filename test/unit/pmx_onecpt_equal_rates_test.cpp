#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/torsten/test/unit/pmx_onecpt_model_test_fixture.hpp>
#include <stan/math/torsten/test/unit/test_util.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using stan::math::var;
using torsten::PMXOneCptModel;

/*
 * ka == k10 = CL/V2 gives the one-compartment system matrix a repeated
 * eigenvalue. The solution is still smooth there: each test compares
 * against the exact confluent limit, written out independently of the
 * solver.
 */
struct TorstenOneCptEqualRatesTest : public testing::Test {
  const double CL = 2.0;
  const double V2 = 10.0;
  const double ka = 0.2;  // == CL / V2 in double precision
  const double k = 0.2;
  const std::vector<double> ts{0.5, 1.0, 4.0, 24.0};

  void SetUp() { stan::math::recover_memory(); }
};

TEST_F(TorstenOneCptEqualRatesTest, rates_are_equal) {
  EXPECT_EQ(CL / V2, ka);
}

TEST_F(TorstenOneCptEqualRatesTest, transient_bolus_and_infusion) {
  const PMXOneCptModel<double> model(CL, V2, ka);
  const std::vector<double> rate{30.0, 7.0};
  for (double t : ts) {
    torsten::PKRec<double> y(2);
    y << 320.0, 50.0;
    model.solve(y, 0.0, t, rate);
    const double e = std::exp(-k * t);
    const double depot = 320.0 * e + rate[0] * (1.0 - e) / k;
    const double central = 50.0 * e + rate[1] * (1.0 - e) / k
                           + 320.0 * k * t * e
                           + rate[0] * (1.0 - e * (1.0 + k * t)) / k;
    EXPECT_NEAR(y(0), depot, 1e-14 * depot) << "t = " << t;
    EXPECT_NEAR(y(1), central, 1e-14 * central) << "t = " << t;
  }
}

TEST_F(TorstenOneCptEqualRatesTest, transient_bolus_gradient) {
  // central amount A0 * ka * D(ka, k10, t), D = (exp(-k10 t) - exp(-ka t)) / (ka - k10);
  // at ka == k10 == k: D = t exp(-k t) and dD/dka = dD/dk10 = -t^2 exp(-k t) / 2.
  const double A0 = 320.0;
  const std::vector<double> rate{0.0, 0.0};
  for (double t : ts) {
    stan::math::recover_memory();
    var CLv = CL, V2v = V2, kav = ka;
    const PMXOneCptModel<var> model(CLv, V2v, kav);
    torsten::PKRec<var> y(2);
    y << A0, 0.0;
    model.solve(y, 0.0, t, rate);
    std::vector<var> theta{CLv, V2v, kav};
    std::vector<double> g;
    y(1).grad(theta, g);

    const double e = std::exp(-k * t);
    const double dD = -0.5 * t * t * e;
    EXPECT_NEAR(y(1).val(), A0 * ka * t * e, 1e-14 * A0 * ka * t * e);
    const double dCL = A0 * ka * dD / V2;
    const double dV2 = -A0 * ka * dD * CL / (V2 * V2);
    const double dka = A0 * (t * e + ka * dD);
    EXPECT_NEAR(g[0], dCL, 1e-13 * std::abs(dCL)) << "t = " << t;
    EXPECT_NEAR(g[1], dV2, 1e-13 * std::abs(dV2)) << "t = " << t;
    EXPECT_NEAR(g[2], dka, 1e-12 * A0 * t) << "t = " << t;
  }
}

TEST_F(TorstenOneCptEqualRatesTest, transient_gradient_vs_finite_diff) {
  const std::vector<double> rate{30.0, 7.0};
  const double t = 4.0;
  auto f1 = [&](const std::vector<double>& pars) {
    const PMXOneCptModel<double> model(pars[0], pars[1], pars[2]);
    torsten::PKRec<double> y(2);
    y << 320.0, 50.0;
    model.solve(y, 0.0, t, rate);
    return y;
  };
  auto f2 = [&](const std::vector<var>& pars) {
    const PMXOneCptModel<var> model(pars[0], pars[1], pars[2]);
    torsten::PKRec<var> y(2);
    y << 320.0, 50.0;
    model.solve(y, 0.0, t, rate);
    return y;
  };
  std::vector<double> pars{CL, V2, ka};
  torsten::test::test_grad(f1, f2, pars, 1.e-3, 1.e-12, 1.e-6, 1.e-8);
}

TEST_F(TorstenOneCptEqualRatesTest, near_equal_rates_are_continuous) {
  // relative separations from ka == k10 down to zero must give values and
  // gradients that converge to the confluent limit, with no cancellation blow-up
  const std::vector<double> rate{30.0, 7.0};
  const double t = 4.0;
  auto solve = [&](double ka_i, std::vector<double>& g) {
    stan::math::recover_memory();
    var CLv = CL, V2v = V2, kav = ka_i;
    const PMXOneCptModel<var> model(CLv, V2v, kav);
    torsten::PKRec<var> y(2);
    y << 320.0, 50.0;
    model.solve(y, 0.0, t, rate);
    std::vector<var> theta{CLv, V2v, kav};
    y(1).grad(theta, g);
    return y(1).val();
  };
  std::vector<double> g0, g;
  const double y0 = solve(ka, g0);
  for (double rel : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14}) {
    for (double sgn : {-1.0, 1.0}) {
      const double y = solve(ka * (1.0 + sgn * rel), g);
      // first-order Taylor in the separation, plus rounding
      const double dy = g0[2] * ka * sgn * rel;
      EXPECT_NEAR(y, y0 + dy, 10 * rel * std::abs(dy) + 1e-14 * y0) << "rel = " << sgn * rel;
      for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(g[i], g0[i], 10 * rel * std::abs(g0[i]) + 1e-12 * std::abs(g0[i]))
          << "rel = " << sgn * rel << ", i = " << i;
      }
    }
  }
}

TEST_F(TorstenOneCptEqualRatesTest, steady_state_bolus) {
  const PMXOneCptModel<double> model(CL, V2, ka);
  const double amt = 1000.0, ii = 12.0;
  const double e = std::exp(-k * ii);
  torsten::PKRec<double> y = model.solve(0.0, amt, 0.0, ii, 1);
  const double depot = amt * e / (1.0 - e);
  const double central = amt * k * ii * e / ((1.0 - e) * (1.0 - e));
  EXPECT_NEAR(y(0), depot, 1e-14 * depot);
  EXPECT_NEAR(y(1), central, 1e-14 * central);

  y = model.solve(0.0, amt, 0.0, ii, 2);
  EXPECT_EQ(y(0), 0.0);
  EXPECT_NEAR(y(1), amt * e / (1.0 - e), 1e-14 * amt);
}

TEST_F(TorstenOneCptEqualRatesTest, steady_state_constant_infusion) {
  const PMXOneCptModel<double> model(CL, V2, ka);
  const double rate = 300.0;
  torsten::PKRec<double> y = model.solve(0.0, 0.0, rate, 0.0, 1);
  EXPECT_NEAR(y(0), rate / ka, 1e-14 * rate / ka);
  EXPECT_NEAR(y(1), rate / k, 1e-14 * rate / k);
}

TEST_F(TorstenOneCptEqualRatesTest, steady_state_infusion_vs_long_run) {
  // truncated infusions, without (rate 300) and with (rate 50) overlap
  // between consecutive infusions, against 400 dosing periods of the
  // transient solution
  const PMXOneCptModel<double> model(CL, V2, ka);
  const double amt = 1000.0, ii = 12.0;
  for (double rate : {300.0, 50.0}) {
    for (int cmt : {1, 2}) {
      torsten::PKRec<double> y_ss = model.solve(0.0, amt, rate, ii, cmt);

      const double t_infus = amt / rate;
      const int n = static_cast<int>(std::floor(t_infus / ii));
      torsten::PKRec<double> y = torsten::PKRec<double>::Zero(2);
      std::vector<double> r1(2, 0.0), r2(2, 0.0);
      r1[cmt - 1] = (n + 1) * rate;
      r2[cmt - 1] = n * rate;
      for (int i = 0; i < 400; ++i) {
        model.solve(y, 0.0, t_infus - n * ii, r1);
        model.solve(y, 0.0, (n + 1) * ii - t_infus, r2);
      }
      EXPECT_NEAR(y_ss(0), y(0), 1e-12 * (1.0 + std::abs(y(0))))
        << "rate = " << rate << ", cmt = " << cmt;
      EXPECT_NEAR(y_ss(1), y(1), 1e-12 * std::abs(y(1)))
        << "rate = " << rate << ", cmt = " << cmt;
    }
  }
}

TEST_F(TorstenOneCptEqualRatesTest, steady_state_gradient_vs_finite_diff) {
  const double amt = 1000.0, ii = 12.0;
  for (double rate : {0.0, 300.0, 50.0}) {
    auto f1 = [&](const std::vector<double>& pars) {
      const PMXOneCptModel<double> model(pars[0], pars[1], pars[2]);
      return model.solve(0.0, amt, rate, ii, 1);
    };
    auto f2 = [&](const std::vector<var>& pars) {
      const PMXOneCptModel<var> model(pars[0], pars[1], pars[2]);
      return model.solve(0.0, amt, rate, ii, 1);
    };
    std::vector<double> pars{CL, V2, ka};
    torsten::test::test_grad(f1, f2, pars, 1.e-3, 1.e-10, 1.e-6, 1.e-8);
  }
}

TEST_F(TorstenOneCptEqualRatesTest, pmx_solve_onecpt_events) {
  // one bolus into the depot, observations in the central compartment
  const std::vector<double> time{0.0, 0.5, 1.0, 4.0, 24.0};
  const std::vector<double> amt{320.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> rate(5, 0.0), ii(5, 0.0);
  const std::vector<int> evid{1, 0, 0, 0, 0}, cmt{1, 2, 2, 2, 2};
  const std::vector<int> addl(5, 0), ss(5, 0);
  const std::vector<double> theta{CL, V2, ka};

  Eigen::MatrixXd x = torsten::pmx_solve_onecpt(time, amt, rate, ii, evid, cmt,
                                                addl, ss, theta);
  for (int i = 1; i < 5; ++i) {
    const double c = 320.0 * ka * time[i] * std::exp(-k * time[i]);
    EXPECT_NEAR(x(1, i), c, 1e-14 * c) << "t = " << time[i];
  }

  std::vector<var> theta_v{CL, V2, ka};
  Eigen::Matrix<var, -1, -1> xv = torsten::pmx_solve_onecpt(time, amt, rate, ii, evid, cmt,
                                                            addl, ss, theta_v);
  std::vector<double> g;
  for (int i = 1; i < 5; ++i) {
    stan::math::set_zero_all_adjoints();
    xv(1, i).grad(theta_v, g);
    const double t = time[i], e = std::exp(-k * t), dD = -0.5 * t * t * e;
    EXPECT_NEAR(xv(1, i).val(), x(1, i), 1e-14 * x(1, i));
    EXPECT_NEAR(g[0], 320.0 * ka * dD / V2, 1e-13 * std::abs(320.0 * ka * dD / V2));
    EXPECT_NEAR(g[2], 320.0 * (t * e + ka * dD), 1e-12 * 320.0 * t);
  }
}
