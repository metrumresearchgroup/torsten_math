#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

using stan::math::var;

/*
 * The analytical compartment models used to solve through
 * eigendecompositions whose entries divide by differences of rates, so
 * they returned NaN where rates coincide: pmx_solve_twocpt where ka equals
 * a disposition rate, pmx_solve_onecpt_effcpt where any two of ka, k10 and
 * ke are equal, and pmx_solve_twocpt_effcpt where ka or ke equals a
 * disposition rate, where ka == ke, and where ka == k21. They now solve
 * through divided differences of exp(-l t) (PMXExpDividedDifferences).
 *
 * Every test compares against an independent long-double reference: the
 * exponential of the system matrix by scaling and squaring, which has no
 * singularity where rates coincide.
 */

namespace {

typedef long double LD;
typedef std::vector<std::vector<LD>> LDMat;

LDMat mul(const LDMat& a, const LDMat& b) {
  const int n = a.size();
  LDMat c(n, std::vector<LD>(n, 0));
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        c[i][j] += a[i][k] * b[k][j];
  return c;
}

/*
 * exp(m) by scaling and squaring of the Taylor series, in long double.
 * The matrices here are Metzler (nonnegative off-diagonal), so the
 * exponential is nonnegative and, once m is scaled below norm 1/4, the
 * series and the squarings lose no significant digits.
 */
LDMat expm(LDMat m) {
  const int n = m.size();
  LD norm = 0;
  for (int j = 0; j < n; ++j) {
    LD c = 0;
    for (int i = 0; i < n; ++i) c += std::abs(m[i][j]);
    norm = std::max(norm, c);
  }
  int s = 0;
  while (norm > 0.25L) {
    norm /= 2;
    ++s;
  }
  for (auto& row : m)
    for (auto& x : row) x = std::ldexp(x, -s);
  LDMat e(n, std::vector<LD>(n, 0)), term(n, std::vector<LD>(n, 0));
  for (int i = 0; i < n; ++i) e[i][i] = term[i][i] = 1;
  for (int k = 1; k <= 30; ++k) {
    term = mul(term, m);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        term[i][j] /= k;
        e[i][j] += term[i][j];
      }
  }
  for (int i = 0; i < s; ++i) e = mul(e, e);
  return e;
}

/*
 * y(t) of y' = A y + r from y(0) = y0: exp(A t) y0 + (integral of exp(A s)
 * over [0, t]) r, from the exponential of the augmented [[A t, r t / c],
 * [0, 0]] with its last column scaled by c, so that a large rate does not
 * raise the norm: the error of scaling and squaring doubles with each
 * squaring.
 */
std::vector<LD> solve_ref(const LDMat& A, const std::vector<LD>& y0,
                          const std::vector<LD>& r, LD t) {
  const int n = A.size();
  LD c = 0;
  for (int i = 0; i < n; ++i) c = std::max(c, std::abs(r[i]) * t);
  if (c == 0) c = 1;
  LDMat m(n + 1, std::vector<LD>(n + 1, 0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) m[i][j] = A[i][j] * t;
    m[i][n] = r[i] * t / c;
  }
  const LDMat e = expm(m);
  std::vector<LD> y(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) y[i] += e[i][j] * y0[j];
    y[i] += c * e[i][n];
  }
  return y;
}

// a x = b by Gaussian elimination with partial pivoting
std::vector<LD> lin_solve(LDMat a, std::vector<LD> b) {
  const int n = b.size();
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(a[i][k]) > std::abs(a[p][k])) p = i;
    std::swap(a[k], a[p]);
    std::swap(b[k], b[p]);
    for (int i = k + 1; i < n; ++i) {
      const LD f = a[i][k] / a[k][k];
      for (int j = k; j < n; ++j) a[i][j] -= f * a[k][j];
      b[i] -= f * b[k];
    }
  }
  std::vector<LD> x(n);
  for (int i = n - 1; i >= 0; --i) {
    LD s = b[i];
    for (int j = i + 1; j < n; ++j) s -= a[i][j] * x[j];
    x[i] = s / a[i][i];
  }
  return x;
}

/*
 * Steady state before the dose of an interval, as the models define it:
 * the periodic solution u = exp(A ii) u + v, v being what one interval's
 * dosing leaves at its end, or u = -A^{-1} r for a constant infusion.
 */
std::vector<LD> steady_state_ref(const LDMat& A, LD amt, LD rate, LD ii, int cmt) {
  const int n = A.size();
  std::vector<LD> zero(n, 0), v(n, 0);
  if (rate == 0 || ii > 0) {
    if (rate == 0) {
      v[cmt - 1] = amt;
      v = solve_ref(A, v, zero, ii);
    } else {
      const LD dt = amt / rate;
      const int k = static_cast<int>(std::floor(dt / ii));
      std::vector<LD> r(n, 0);
      r[cmt - 1] = (k + 1) * rate;
      v = solve_ref(A, v, r, dt - k * ii);
      r[cmt - 1] = k * rate;
      v = solve_ref(A, v, r, (k + 1) * ii - dt);
    }
    LDMat m = expm([&] {
      LDMat a = A;
      for (auto& row : a)
        for (auto& x : row) x *= ii;
      return a;
    }());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) m[i][j] = (i == j) - m[i][j];
    return lin_solve(m, v);
  }
  LDMat a = A;
  for (auto& row : a)
    for (auto& x : row) x = -x;
  std::vector<LD> r(n, 0);
  r[cmt - 1] = rate;
  return lin_solve(a, r);
}

/*
 * Divided difference g[-l_0, ..., -l_n] of g(x) = exp(x t): by Opitz's
 * formula the (n, 0) entry of exp(J t), J lower bidiagonal with diagonal
 * -l_i and unit subdiagonal.
 */
LD dd_ref(const std::vector<LD>& l, LD t) {
  const int n = l.size();
  LDMat m(n, std::vector<LD>(n, 0));
  for (int i = 0; i < n; ++i) {
    m[i][i] = -l[i] * t;
    if (i > 0) m[i][i - 1] = t;
  }
  return expm(m)[n - 1][0];
}

// system matrices in long double, from the models' parameters
LDMat twocpt_matrix(const std::vector<LD>& p) {  // CL, Q, V2, V3, ka [, ke]
  const LD k10 = p[0] / p[2], k12 = p[1] / p[2], k21 = p[1] / p[3];
  const int n = p.size() == 6 ? 4 : 3;
  LDMat A(n, std::vector<LD>(n, 0));
  A[0][0] = -p[4];
  A[1][0] = p[4];
  A[1][1] = -(k10 + k12);
  A[1][2] = k21;
  A[2][1] = k12;
  A[2][2] = -k21;
  if (n == 4) {
    A[3][1] = p[5];
    A[3][3] = -p[5];
  }
  return A;
}

LDMat onecpt_effcpt_matrix(const std::vector<LD>& p) {  // CL, V, ka, ke
  LDMat A(3, std::vector<LD>(3, 0));
  A[0][0] = -p[2];
  A[1][0] = p[2];
  A[1][1] = -p[0] / p[1];
  A[2][1] = p[3];
  A[2][2] = -p[3];
  return A;
}

std::vector<LD> to_ld(const std::vector<double>& x) { return std::vector<LD>(x.begin(), x.end()); }

template <typename T>
std::string str(const std::vector<T>& x) {
  std::string s;
  for (const auto& xi : x) s += std::to_string(static_cast<double>(xi)) + " ";
  return s;
}

// relative error per entry: every amount is a positive combination of
// positive terms, so each entry is accurate relative to itself
template <typename V>
void expect_near_ref(const V& y, const std::vector<LD>& ref, double rtol, const std::string& what) {
  ASSERT_EQ(static_cast<size_t>(y.size()), ref.size()) << what;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double yi = stan::math::value_of(y(i));
    EXPECT_TRUE(std::isfinite(yi)) << what << ", cmt " << i + 1;
    EXPECT_NEAR(yi, static_cast<double>(ref[i]), rtol * std::abs(static_cast<double>(ref[i])) + 1e-300)
      << what << ", cmt " << i + 1;
  }
}

/*
 * AD gradients of every output of f(p) against long-double central
 * differences of the reference f_ref(p).
 */
void expect_grad_near_ref(const std::vector<double>& p,
                          const std::function<Eigen::Matrix<var, -1, 1>(const std::vector<var>&)>& f,
                          const std::function<std::vector<LD>(const std::vector<LD>&)>& f_ref,
                          double rtol, const std::string& what) {
  stan::math::recover_memory();
  const std::vector<var> pv(p.begin(), p.end());
  Eigen::Matrix<var, -1, 1> y = f(pv);
  const std::vector<LD> y_ref = f_ref(to_ld(p));
  std::vector<std::vector<LD>> d(p.size());
  for (size_t j = 0; j < p.size(); ++j) {
    std::vector<LD> p1 = to_ld(p), p2 = to_ld(p);
    const LD h = 1e-6L * std::max<LD>(1, std::abs(p1[j]));
    p1[j] -= h;
    p2[j] += h;
    const std::vector<LD> y1 = f_ref(p1), y2 = f_ref(p2);
    for (size_t i = 0; i < y1.size(); ++i) d[j].push_back((y2[i] - y1[i]) / (2 * h));
  }
  for (int i = 0; i < y.size(); ++i) {
    stan::math::set_zero_all_adjoints();
    y(i).grad();
    for (size_t j = 0; j < p.size(); ++j) {
      const double g = pv[j].adj(), g_ref = static_cast<double>(d[j][i]);
      EXPECT_TRUE(std::isfinite(g)) << what << ", output " << i << ", parameter " << j;
      EXPECT_NEAR(g, g_ref, rtol * std::abs(g_ref) + 1e-11 * (1 + std::abs(static_cast<double>(y_ref[i]))))
        << what << ", output " << i << ", parameter " << j;
    }
  }
}

// index of ka in a model's parameters
template <template <typename> class M>
constexpr int ka_index() { return M<double>::Npar == 4 ? 2 : 4; }

/*
 * A model with the parameter sets that put rates on top of each other.
 * The two-compartment parameters CL = 5, Q = 2.5, V2 = 10, V3 = 5 give
 * k10 = 0.5, k12 = 0.25, k21 = 0.5 and the disposition rates exactly 1
 * and 0.25, also in double precision; the one-compartment parameters
 * CL = 2, V = 10 give k10 = 0.2.
 */
template <template <typename> class M>
struct Case {
  std::string name;
  std::vector<std::vector<double>> pars;  // labelled by the coincidence they hit
  std::vector<std::string> labels;
  std::function<LDMat(const std::vector<LD>&)> matrix;
  int ss_max_cmt;
};

Case<torsten::PMXTwoCptModel> twocpt_case() {
  return {"twocpt",
          {{5, 2.5, 10, 5, 1.0}, {5, 2.5, 10, 5, 0.25}, {5, 2.5, 10, 5, 1.0 + 1e-8},
           {5, 2.5, 10, 5, 0.25 * (1 - 1e-12)}, {5, 2.5, 10, 5, 0.7}, {5, 2.5, 10, 5, 0.0}},
          {"ka == alpha", "ka == beta", "ka - alpha = 1e-8", "ka / beta = 1 - 1e-12",
           "distinct rates", "ka == 0"},
          twocpt_matrix, 3};
}

Case<torsten::PMXOneCptEffCptModel> onecpt_effcpt_case() {
  return {"onecpt_effcpt",
          {{2, 10, 0.2, 1.0}, {2, 10, 1.0, 0.2}, {2, 10, 1.0, 1.0}, {2, 10, 0.2, 0.2},
           {2, 10, 0.2 * (1 + 1e-10), 0.2 * (1 - 1e-10)}, {2, 10, 0.7, 1.0}, {2, 10, 0.0, 0.2}},
          {"ka == k10", "ke == k10", "ka == ke", "ka == k10 == ke", "all within 1e-10",
           "distinct rates", "ka == 0, ke == k10"},
          onecpt_effcpt_matrix, 3};
}

Case<torsten::PMXTwocptEffCptModel> twocpt_effcpt_case() {
  return {"twocpt_effcpt",
          {{5, 2.5, 10, 5, 1.0, 2.0}, {5, 2.5, 10, 5, 0.25, 2.0}, {5, 2.5, 10, 5, 0.5, 2.0},
           {5, 2.5, 10, 5, 2.0, 2.0}, {5, 2.5, 10, 5, 0.7, 1.0}, {5, 2.5, 10, 5, 0.7, 0.25},
           {5, 2.5, 10, 5, 1.0, 1.0}, {5, 2.5, 10, 5, 0.25, 1.0}, {5, 2.5, 10, 5, 0.7, 2.0},
           {5, 2.5, 10, 5, 0.0, 0.25}},
          {"ka == alpha", "ka == beta", "ka == k21", "ka == ke", "ke == alpha", "ke == beta",
           "ka == ke == alpha", "ka == beta, ke == alpha", "distinct rates", "ka == 0, ke == beta"},
          twocpt_matrix, 3};
}

template <template <typename> class M>
void check_transient(const Case<M>& c) {
  for (size_t k = 0; k < c.pars.size(); ++k) {
    const std::vector<double>& p = c.pars[k];
    const M<double> model(p);
    const LDMat A = c.matrix(to_ld(p));
    const int n = M<double>::Ncmt;
    for (double t : {0.5, 4.0, 24.0}) {
      for (int src = 0; src < n; ++src) {
        for (bool infusion : {false, true}) {
          std::vector<double> y0(n, 0.0), rate(n, 0.0);
          (infusion ? rate : y0)[src] = infusion ? 7.0 : 100.0;
          torsten::PKRec<double> y = Eigen::Map<Eigen::VectorXd>(y0.data(), n);
          model.solve(y, 0.0, t, rate);
          expect_near_ref(y, solve_ref(A, to_ld(y0), to_ld(rate), t), 1e-13,
                          c.name + " " + c.labels[k] + (infusion ? ", infusion" : ", bolus")
                          + " into cmt " + std::to_string(src + 1) + ", t = " + std::to_string(t));
        }
      }
    }
  }
}

template <template <typename> class M>
void check_transient_grad(const Case<M>& c) {
  for (size_t k = 0; k < c.pars.size(); ++k) {
    const int n = M<double>::Ncmt;
    std::vector<double> y0{100.0, 20.0, 10.0, 5.0}, rate{7.0, 3.0, 2.0, 1.0};
    y0.resize(n);
    rate.resize(n);
    const double t = 4.0;
    auto f = [&](const std::vector<var>& pv) {
      const M<var> model(pv);
      torsten::PKRec<var> y = Eigen::Map<Eigen::VectorXd>(y0.data(), n).template cast<var>();
      model.solve(y, 0.0, t, rate);
      return y;
    };
    auto f_ref = [&](const std::vector<LD>& p) { return solve_ref(c.matrix(p), to_ld(y0), to_ld(rate), t); };
    expect_grad_near_ref(c.pars[k], f, f_ref, 1e-8, c.name + " " + c.labels[k]);
  }
}

template <template <typename> class M>
void check_steady_state(const Case<M>& c) {
  const double amt = 1000.0, ii = 12.0;
  for (size_t k = 0; k < c.pars.size(); ++k) {
    const std::vector<double>& p = c.pars[k];
    if (p[ka_index<M>()] == 0.0) continue;  // steady state needs ka > 0
    const M<double> model(p);
    const LDMat A = c.matrix(to_ld(p));
    // bolus; infusions shorter than ii and overlapping over three
    // intervals; constant infusion
    const std::vector<std::vector<double>> doses{{amt, 0.0, ii}, {amt, amt / 3, ii},
                                                 {amt, amt / 30, ii}, {0.0, 300.0, 0.0}};
    for (int cmt = 1; cmt <= c.ss_max_cmt; ++cmt) {
      for (const auto& d : doses) {
        const torsten::PKRec<double> u = model.solve(0.0, d[0], d[1], d[2], cmt);
        expect_near_ref(u, steady_state_ref(A, d[0], d[1], d[2], cmt), 1e-12,
                        c.name + " " + c.labels[k] + ", steady state into cmt " + std::to_string(cmt)
                        + ", dose " + str(d));
      }
    }
  }
}

template <template <typename> class M>
void check_steady_state_grad(const Case<M>& c) {
  const double amt = 1000.0, ii = 12.0;
  for (size_t k = 0; k < c.pars.size(); ++k) {
    if (c.pars[k][ka_index<M>()] == 0.0) continue;
    for (double rate : {0.0, amt / 30}) {
      auto f = [&](const std::vector<var>& pv) {
        const M<var> model(pv);
        return model.solve(0.0, amt, rate, ii, 1);
      };
      auto f_ref = [&](const std::vector<LD>& p) { return steady_state_ref(c.matrix(p), amt, rate, ii, 1); };
      expect_grad_near_ref(c.pars[k], f, f_ref, 1e-7, c.name + " " + c.labels[k] + ", rate " + std::to_string(rate));
    }
  }
}

}  // namespace

TEST(TorstenExpDividedDifferences, confluent_closed_forms) {
  using torsten::internal::PMXExpDividedDifferences;
  // G({a x (n + 1)}) = t^n exp(-a t) / n! and G({0, a x n}) = P(n, a t) / a^n,
  // with P(n, x) = exp(-x) sum_{k >= n} x^k / k! the regularized lower
  // incomplete gamma function, summed in long double without cancellation
  auto gamma_p = [](int n, LD x) {
    LD term = std::exp(-x), sum = 0;
    for (int k = 1; k <= n; ++k) term *= x / k;
    for (int k = n; k < n + 200; ++k) {
      sum += term;
      term *= x / (k + 1);
    }
    return sum;
  };
  auto expect_rel = [](double y, LD ref, const std::string& what) {
    EXPECT_NEAR(y, static_cast<double>(ref), 1e-15 * std::abs(static_cast<double>(ref))) << what;
  };
  const double a = 0.3;
  for (double t : {0.0, 0.1, 2.0, 30.0}) {
    PMXExpDividedDifferences<double, 4> g(std::array<double, 4>{0.0, a, a, a}, t);
    const LD e = std::exp(-static_cast<LD>(a) * t), at = static_cast<LD>(a) * t;
    const std::string where = "t = " + std::to_string(t);
    EXPECT_DOUBLE_EQ(g(2), std::exp(-a * t));
    expect_rel(g(2 | 4), t * e, where);
    expect_rel(g(2 | 4 | 8), t * t * e / 2, where);
    expect_rel(g(1 | 2), gamma_p(1, at) / a, where);
    expect_rel(g(1 | 2 | 4), gamma_p(2, at) / (a * a), where);
    expect_rel(g(1 | 2 | 4 | 8), gamma_p(3, at) / (a * a * a), where);
    PMXExpDividedDifferences<double, 3> z(std::array<double, 3>{0.0, 0.0, 0.0}, t);
    EXPECT_EQ(z(1), 1.0);
    expect_rel(z(1 | 2), t, where);
    expect_rel(z(1 | 2 | 4), static_cast<LD>(t) * t / 2, where);
  }
}

TEST(TorstenExpDividedDifferences, against_bidiagonal_exponential) {
  // separations of the rates from 0 up to wide, on both sides of the switch
  // between the Taylor series and the recurrence at (l_max - l_min) t = 1
  using torsten::internal::PMXExpDividedDifferences;
  const std::vector<double> sep{0.0, 1e-14, 1e-9, 1e-5, 1e-2, 0.2, 0.9, 1.1, 3.0, 20.0};
  for (double t : {0.5, 4.0, 24.0}) {
    for (double s1 : sep) {
      for (double s2 : sep) {
        const std::array<double, 4> l{0.0, 0.4, 0.4 + s1 / t, 0.4 + (s1 + s2) / t};
        PMXExpDividedDifferences<double, 4> g(l, t);
        for (unsigned s = 1; s < 16; ++s) {
          std::vector<LD> ls;
          for (int i = 0; i < 4; ++i)
            if (s & (1u << i)) ls.push_back(l[i]);
          const double ref = static_cast<double>(dd_ref(ls, t));
          EXPECT_NEAR(g(s), ref, 1e-14 * ref) << "rates " << str(ls) << "t = " << t;
        }
      }
    }
  }
}

TEST(TorstenExpDividedDifferences, gradient_is_divided_difference_with_repeated_rate) {
  // d G(S) / d l_i = -G(S + {l_i}), whether or not rates coincide
  using torsten::internal::PMXExpDividedDifferences;
  const double t = 4.0;
  for (double s : {0.0, 1e-12, 1e-6, 0.1, 0.3, 2.0}) {
    stan::math::recover_memory();
    std::vector<var> l{0.0, 0.5, 0.5 + s, 0.5 + 2 * s};
    PMXExpDividedDifferences<var, 4> g(std::array<var, 4>{l[0], l[1], l[2], l[3]}, t);
    var y = g(15);
    y.grad();
    for (int i = 0; i < 4; ++i) {
      std::vector<LD> ls{l[0].val(), l[1].val(), l[2].val(), l[3].val(), l[i].val()};
      const double ref = -static_cast<double>(dd_ref(ls, t));
      EXPECT_NEAR(l[i].adj(), ref, 1e-13 * std::abs(ref)) << "separation " << s << ", rate " << i;
    }
  }
}

TEST(TorstenCptEqualRates, transient) {
  check_transient(twocpt_case());
  check_transient(onecpt_effcpt_case());
  check_transient(twocpt_effcpt_case());
}

TEST(TorstenCptEqualRates, transient_gradient) {
  check_transient_grad(twocpt_case());
  check_transient_grad(onecpt_effcpt_case());
  check_transient_grad(twocpt_effcpt_case());
}

TEST(TorstenCptEqualRates, steady_state) {
  // includes distinct rates: before, pmx_solve_onecpt_effcpt's steady state
  // was wrong in the effect compartment even there
  check_steady_state(twocpt_case());
  check_steady_state(onecpt_effcpt_case());
  check_steady_state(twocpt_effcpt_case());
}

TEST(TorstenCptEqualRates, steady_state_gradient) {
  check_steady_state_grad(twocpt_case());
  check_steady_state_grad(onecpt_effcpt_case());
  check_steady_state_grad(twocpt_effcpt_case());
}

TEST(TorstenCptEqualRates, pmx_solve_events) {
  // a bolus into the depot at 0, a 2 h infusion into the central compartment
  // at 1, observations after; ka on a disposition rate (and ke too)
  const std::vector<double> time{0.0, 0.5, 1.0, 2.0, 4.0, 12.0, 24.0};
  const std::vector<double> amt{100.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> rate{0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> ii(7, 0.0);
  const std::vector<int> evid{1, 0, 1, 0, 0, 0, 0}, cmt{1, 2, 2, 2, 2, 2, 2};
  const std::vector<int> addl(7, 0), ss(7, 0);

  // amounts after each event, stepping the reference through the event table
  auto ref = [&](const LDMat& A) {
    const int n = A.size();
    std::vector<std::vector<LD>> out;
    std::vector<LD> y(n, 0), r(n, 0);
    LD t = 0;
    for (size_t i = 0; i < time.size(); ++i) {
      if (time[i] > 3.0 && t < 3.0) {  // the infusion ends at 3
        y = solve_ref(A, y, r, 3.0L - t);
        t = 3.0;
        r[1] = 0;
      }
      y = solve_ref(A, y, r, time[i] - t);
      t = time[i];
      if (evid[i] == 1 && rate[i] == 0) y[cmt[i] - 1] += amt[i];
      if (evid[i] == 1 && rate[i] > 0) r[cmt[i] - 1] += rate[i];
      out.push_back(y);
    }
    return out;
  };
  auto check = [&](const std::string& name, const std::vector<double>& theta, const LDMat& A,
                   auto solver) {
    const Eigen::MatrixXd x = solver(time, amt, rate, ii, evid, cmt, addl, ss, theta);
    const std::vector<std::vector<LD>> y = ref(A);
    for (size_t i = 0; i < time.size(); ++i) {
      if (evid[i] == 0) {
        expect_near_ref(x.col(i), y[i], 1e-13, name + " at t = " + std::to_string(time[i]));
      }
    }
    std::vector<var> theta_v(theta.begin(), theta.end());
    Eigen::Matrix<var, -1, -1> xv = solver(time, amt, rate, ii, evid, cmt, addl, ss, theta_v);
    std::vector<double> g;
    for (int i = 0; i < xv.size(); ++i) {
      stan::math::set_zero_all_adjoints();
      xv(i).grad(theta_v, g);
      for (double gi : g) EXPECT_TRUE(std::isfinite(gi)) << name;
    }
  };
  check("pmx_solve_twocpt, ka == alpha", {5, 2.5, 10, 5, 1.0}, twocpt_matrix({5, 2.5, 10, 5, 1.0}),
        [](auto&&... a) { return torsten::pmx_solve_twocpt(a...); });
  check("pmx_solve_onecpt_effcpt, ka == k10 == ke", {2, 10, 0.2, 0.2},
        onecpt_effcpt_matrix({2, 10, 0.2, 0.2}),
        [](auto&&... a) { return torsten::pmx_solve_onecpt_effcpt(a...); });
  check("pmx_solve_twocpt_effcpt, ka == beta, ke == alpha", {5, 2.5, 10, 5, 0.25, 1.0},
        twocpt_matrix({5, 2.5, 10, 5, 0.25, 1.0}),
        [](auto&&... a) { return torsten::pmx_solve_twocpt_effcpt(a...); });
}
